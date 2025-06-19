# torch_port.py  –  PyTorch translation of your TensorFlow models
# --------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch import optim

# ============ 1)  Basic building blocks =======================

class Generator(nn.Module):
    """
    Dense generator that concatenates uniform noise with a conditioning vector.
    """
    def __init__(self, output_shape, latent_dim, layer_sizes=None):
        super().__init__()
        layer_sizes = layer_sizes or []
        self.output_shape = tuple(output_shape)
        self.latent_dim   = latent_dim

        net = []
        in_dim = latent_dim                      # `y` is expected to be (B, latent_dim)
        for i, units in enumerate(layer_sizes):
            net.append(nn.Linear(in_dim, units))
            net.append(nn.Sigmoid())
            in_dim = units
        net.append(nn.Linear(in_dim, math.prod(output_shape)))
        self.net = nn.Sequential(*net)

    def forward(self, y):
        z = torch.rand_like(y)
        x = torch.cat([y, z], dim=-1)
        x = self.net(x)
        return x.view(-1, *self.output_shape)


class DistributionGenerator(nn.Module):
    """
    Dense generator that outputs μ and σ, then re-parametrises.
    """
    def __init__(self, output_shape, latent_dim, layer_sizes=(256,512,1024)):
        super().__init__()
        self.output_shape = tuple(output_shape)

        layers      = []
        in_dim      = latent_dim
        for units in layer_sizes:
            layers += [nn.Linear(in_dim, units), nn.ReLU(inplace=True)]
            in_dim = units
        self.backbone = nn.Sequential(*layers)

        self.mu_head    = nn.Linear(in_dim, math.prod(output_shape))
        self.sigma_head = nn.Linear(in_dim, math.prod(output_shape))

    def forward(self, y):
        h   = self.backbone(y)
        mu  = self.mu_head(h)
        sig = F.softplus(self.sigma_head(h)) + 1e-6
        eps = torch.randn_like(mu)
        x   = mu + sig * eps
        return x.view(-1, *self.output_shape)


class ConvDistributionGenerator(nn.Module):
    """
    Dense → 3-D deconvolutional generator with residual re-parametrised blocks.
    Matches the Keras version layer-for-layer.
    """
    def __init__(self, output_shape, latent_dim, layer_sizes=(256,512,1024),
                 conv_filters=(16,8,1), stride=(1,1,1), kernel_size=(5,5,5)):
        super().__init__()
        D,H,W = output_shape
        self.D, self.H, self.W = D, H, W
        self.output_shape = output_shape

        # tail: dense backbone
        tail, in_dim = [], latent_dim
        for u in layer_sizes:
            tail += [nn.Linear(in_dim, u), nn.ReLU(inplace=True)]
            in_dim = u
        self.tail = nn.Sequential(*tail)

        # compute shape fed into first transposed-conv
        self.num_conv = len(conv_filters)
        d, h, w = D, H, W
        for _ in range(self.num_conv):
            d //= stride[0]; h //= stride[1]; w //= stride[2]
        self.flat_sz = d*h*w

        self.fc_to_vox = nn.Sequential(
            nn.Linear(in_dim, self.flat_sz),
            nn.ReLU(inplace=True)
        )

        # build conv blocks
        blocks = nn.ModuleList()
        cur_ch = 1
        for idx, out_ch in enumerate(conv_filters):
            deconv = nn.ConvTranspose3d(cur_ch, out_ch,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=(kernel_size[0]//2, kernel_size[1]//2, kernel_size[2]//2))
            mu_fc     = nn.Linear(in_dim, self.flat_sz)
            sigma_fc  = nn.Linear(in_dim, self.flat_sz)
            blocks.append(nn.ModuleDict({
                "deconv": deconv, "mu": mu_fc, "sigma": sigma_fc
            }))
            cur_ch = out_ch
        self.blocks = blocks
        self.reshape = lambda t: t.view(t.size(0), 1, d, h, w)  # helper

    def forward(self, y):
        latent = self.tail(y)
        vox    = self.reshape(self.fc_to_vox(latent))
        for blk in self.blocks:
            vox = blk["deconv"](vox)
            mu  = blk["mu"](latent)
            sig = F.softplus(blk["sigma"](latent)) + 1e-6
            eps = torch.randn_like(mu)
            res = F.relu(mu + sig*eps).view(eps.size(0), 1, self.D, self.H, self.W)

            vox = vox + res

        # final reshape to (B, D*H, W) to match TF code
        B, C, D, H, W = vox.shape
        return vox.view(B, D*H, W)


class Autoencoder(nn.Module):
    """
    Standard dense auto-encoder (encoder+decoder kept together).
    """
    def __init__(self, input_shape, latent_dim,
                 encoder_layer_sizes=(), decoder_layer_sizes=()):
        super().__init__()
        in_dim = math.prod(input_shape)
        enc = [nn.Flatten()]
        for u in encoder_layer_sizes:
            enc += [nn.Linear(in_dim, u), nn.ReLU(inplace=True)]
            in_dim = u
        enc.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc)

        dec = []
        in_dim = latent_dim
        for u in decoder_layer_sizes:
            dec += [nn.Linear(in_dim, u), nn.ReLU(inplace=True)]
            in_dim = u
        dec += [nn.Linear(in_dim, math.prod(input_shape)),
                nn.Softplus(),
                nn.Unflatten(1, input_shape)]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):              # alias kept for trainer
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


class MinibatchDiscrimination(nn.Module):
    """
    Exact functional match to the TF version (L1 distance kernel).
    """
    def __init__(self, num_kernels, kernel_dim):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernel_dim  = kernel_dim
        self.T = nn.Parameter(torch.randn(0))   # init in `reset_parameters`

    def reset_parameters(self, in_features):
        self.T = nn.Parameter(
            torch.empty(in_features, self.num_kernels * self.kernel_dim)
        )
        nn.init.xavier_uniform_(self.T)

    def forward(self, x):
        if self.T.numel() == 0:        # lazy init to know in_features
            self.reset_parameters(x.shape[-1])
            # Ensure T is on the same device as x
            self.T = nn.Parameter(self.T.to(x.device))

        M = x @ self.T                 # (B, num_kernels*kernel_dim)
        M = M.view(-1, self.num_kernels, self.kernel_dim)

        M1 = M.unsqueeze(0)            # (1,B, K, D)
        M2 = M.unsqueeze(1)            # (B,1, K, D)
        l1 = (M1 - M2).abs().sum(dim=3)  # (B,B,K)
        K = torch.exp(-l1)
        return K.sum(dim=1) - 1        # (B,K)


# ============ 2)  Discriminators =================================

class Discriminator(nn.Module):
    """
    Dense critic with optional feature extraction (Salimans et al. 2016).
    """
    def __init__(self, input_shape, layer_sizes=(256,128)):
        super().__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            *sum([[nn.Linear(in_f := (math.prod(input_shape) if i==0 else layer_sizes[i-1]), u),
                   nn.ReLU(inplace=True)] for i, u in enumerate(layer_sizes)], [])
        )

        self.condition = nn.Sequential(
            nn.Flatten(),
            nn.Linear(      0, 64),   # placeholder, fixed in `reset_cond`
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self._cond_reset_needed = True
        self.joint = nn.Sequential(
            nn.Linear(layer_sizes[-1] + 32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.mbd  = MinibatchDiscrimination(32, 16)
        self.final = nn.Linear(32 + 32, 1)

    def _maybe_reset_condition_layers(self, cond):
        if self._cond_reset_needed:
            in_f = cond.view(cond.size(0), -1).size(1)
            self.condition[0] = nn.Linear(in_f, 64)
            self._cond_reset_needed = False

    def _forward_base(self, x, cond):
        x = self.main(x)
        self._maybe_reset_condition_layers(cond)
        cond = self.condition(cond)
        h = torch.cat([x, cond], dim=-1)
        h = self.joint(h)
        return h

    def forward(self, inputs):
        x, cond = inputs
        feat = self._forward_base(x, cond)
        enhanced = torch.cat([feat, self.mbd(feat)], dim=-1)
        return self.final(enhanced)

    def extract_features(self, inputs):
        x, cond = inputs
        return self._forward_base(x, cond)


class Conv3DDiscriminator(nn.Module):
    """
    3-D convolutional critic for (B,24,24,700) grids.
    """
    def __init__(self, input_shape=(24,24,700), input_cond_shape=(1,4,4)):
        super().__init__()
        D,H,W = input_shape
        self.D, self.H, self.W = D, H, W
        self.image_branch = nn.Sequential(
            nn.Conv3d(1,  64, (3,3,7), stride=(1,1,4), padding='valid'),
            nn.ReLU(inplace=True),
            nn.Conv3d(64,128, (3,3,5), stride=(1,1,2), padding='valid'),
            nn.ReLU(inplace=True),
            nn.Conv3d(128,256,(3,3,3), stride=1,            padding='valid'),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),                        # (B,256,1,1,1)
            nn.Flatten(),                                   # (B,256)
        )

        self.cond = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self._cond_reset_needed = True

        self.joint = nn.Sequential(
            nn.Linear(256+32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.mbd   = MinibatchDiscrimination(32, 16)
        self.final = nn.Linear(32+32, 1)

    def _forward_base(self, x, cond):
        x = x.view(x.size(0), 1, self.D, self.H, self.W)
        img_feat = self.image_branch(x)
        cond = cond.view(cond.size(0), -1)
        cond_feat = self.cond(cond)
        h = torch.cat([img_feat, cond_feat], dim=-1)
        return self.joint(h)

    def forward(self, inputs):
        x, cond = inputs
        feat = self._forward_base(x, cond)
        enhanced = torch.cat([feat, self.mbd(feat)], dim=-1)
        return self.final(enhanced)

    def extract_features(self, inputs):
        x, cond = inputs
        return self._forward_base(x, cond)


# ============ 3)  WGAN-GP trainer with feature-matching =========

class GANTrainer:
    """
    PyTorch re-implementation of your Keras GANTrainer.
    Works with any (encoder, generator, discriminator) triple
    conforming to the APIs above.
    """
    def __init__(self,
                 autoencoder, generator, discriminator,
                 *,
                 reconstruction_weight=0.0,
                 generator_weight=1.0,
                 feature_matching_weight=10.0,
                 critic_steps=5, gp_weight=10.0,
                 device='mps' if torch.backends.mps.is_available() else 'cpu'):

        self.device = device
        self.cpu_device = 'cpu'
        
        # Place autoencoder and discriminator on MPS (or whatever device is specified)
        self.ae = autoencoder.to(self.device).eval()      # encoder frozen
        self.disc = discriminator.to(self.device)
        
        # Place generator on CPU explicitly
        self.gen = generator.to(self.cpu_device)

        self.recon_w = reconstruction_weight
        self.gen_w = generator_weight
        self.fm_w = feature_matching_weight
        self.critic_steps = critic_steps
        self.gp_w = gp_weight

        self.opt_gen = optim.RMSprop(self.gen.parameters(), 5e-5)
        self.opt_disc = optim.RMSprop(self.disc.parameters(), 5e-5)

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    def _gradient_penalty(self, real_x, fake_x, cond_y):
        alpha = torch.rand(real_x.size(0), 1, 1, device=self.device)
        inter = real_x + alpha*(fake_x-real_x)
        inter.requires_grad_(True)
        score = self.disc([inter, cond_y]).sum()

        grads = autograd.grad(score, inter, create_graph=True)[0]
        grads = grads.view(grads.size(0), -1)
        gp = ((grads.norm(2, dim=1) - 1)**2).mean()
        return gp

    # ------------------------------------------------------------
    def train_step(self, real_x, cond_y):
        real_x, cond_y = real_x.to(self.device), cond_y.to(self.device)

        with torch.no_grad():
            z = self.ae.encode(real_x.view(real_x.size(0), -1))
            # Move latent vectors to CPU for generator
            z_cpu = z.to(self.cpu_device)

        # ------- 1) critic ---------
        for _ in range(self.critic_steps):
            # Generate on CPU then move to device for discriminator
            fake_x_cpu = self.gen(z_cpu).detach()
            fake_x = fake_x_cpu.to(self.device)
            
            fake_score = self.disc([fake_x, cond_y]).mean()
            real_score = self.disc([real_x, cond_y]).mean()
            gp = self._gradient_penalty(real_x, fake_x, cond_y)
            disc_loss = fake_score - real_score + self.gp_w*gp

            self.opt_disc.zero_grad()
            disc_loss.backward()
            self.opt_disc.step()

        # ------- 2) generator ------
        # Generate on CPU
        fake_x_cpu = self.gen(z_cpu)
        # Move to device for discriminator
        fake_x = fake_x_cpu.to(self.device)
        
        fake_score = self.disc([fake_x, cond_y]).mean()
        gen_loss_gan = -fake_score

        fm_loss = torch.zeros(1, device=self.device)
        if self.fm_w > 0:
            real_feat = self.disc.extract_features([real_x, cond_y]).mean(0)
            fake_feat = self.disc.extract_features([fake_x, cond_y]).mean(0)
            fm_loss = F.mse_loss(real_feat, fake_feat)

        gen_loss = self.gen_w*gen_loss_gan + self.fm_w*fm_loss

        # Move loss to CPU for generator backward pass
        gen_loss_cpu = gen_loss.to(self.cpu_device)
        
        self.opt_gen.zero_grad()
        gen_loss_cpu.backward()
        self.opt_gen.step()

        return {
            "disc_loss":     disc_loss.item(),
            "gen_loss":      gen_loss_gan.item(),
            "fm_loss":       fm_loss.item(),
            "wasserstein":   (real_score - fake_score).item(),
            "gp":            gp.item(),
        }
