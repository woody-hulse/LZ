# train_dense_gan_torch.py
# ---------------------------------------------------------------
import os, time, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import MaxNLocator

from gan_torch import (Autoencoder,
                        ConvDistributionGenerator,
                        Conv3DDiscriminator,
                        GANTrainer)

# --- domain-specific helpers (unchanged, still NumPy / TF-agnostic) ----
from simple_pulse import (Params,
                          vertex_electron_batch_generator,
                          generate_vertex_electron_params,
                          simulate_fixed_vertex_electron)
from autoencoder_analysis import plot_3d_scatter_with_profiles, plot_events
from train_gan import generate_and_save_sample
# -----------------------------------------------------------------------


# ======================================================================
# 0)  pretty plotting helper  (unchanged except for TF-free interface)
# ======================================================================
def plot_training_metrics(metrics_history, save_path=None, window_size=20):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14), dpi=150, sharex=True)

    epochs = np.arange(len(metrics_history['generator_loss']))

    colors = dict(
        wasserstein_distance='#0000ff',
        gradient_penalty     ='#4c72b0',
        generator_loss       ='#ffa500',
        critic_loss          ='#0000ff',
        reconstruction_loss  ='#2ca02c',
        critic_accuracy      ='#8b0000',
        generator_fool_rate  ='#ff4500',
    )

    # --- (1) distance & GP -------------------------------------------
    ax1.plot(epochs, metrics_history['wasserstein_distance'],
             color=colors['wasserstein_distance'], lw=2, label='Wasserstein D')
    ax1.plot(epochs, metrics_history['gradient_penalty'],
             color=colors['gradient_penalty'], lw=2, alpha=.7, label='Gradient Penalty')
    ax1.set_ylabel('Distance / Penalty'), ax1.legend(), ax1.grid(alpha=.3)
    ax1.set_title('WGAN Training Metrics')

    # --- (2) losses ---------------------------------------------------
    for m in ['generator_loss', 'critic_loss']:
        vals = np.asarray(metrics_history[m])
        ax2.scatter(epochs, vals, s=15, alpha=.2, color=colors[m])
        ax2.plot(epochs, gaussian_filter1d(vals, sigma=window_size/3),
                 lw=1.5, color=colors[m], label=m.replace('_',' ').title())
    ax2.set_ylabel('Loss'), ax2.grid(alpha=.3), ax2.legend()
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- (3) misc -----------------------------------------------------
    for m in ['critic_accuracy', 'generator_fool_rate']:
        vals = np.asarray(metrics_history[m])
        ax3.scatter(epochs, vals, s=15, alpha=.2, color=colors[m])
        ax3.plot(epochs, gaussian_filter1d(vals, sigma=window_size/3),
                 lw=1.5, color=colors[m], label=m.replace('_',' ').title())
    ax3.set_xlabel('Epoch'), ax3.set_ylabel('Metric')
    ax3.grid(alpha=.3), ax3.legend()
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {save_path}')
    return fig


# ======================================================================
# 1)  main training routine (PyTorch)
# ======================================================================
def train_dense_gan(*,
    # --- model / data --------------------------------------------------
    input_shape=(24,24,700),
    latent_dim=16,
    encoder_layer_sizes=(32,64,128),
    decoder_layer_sizes=(128,64,32),
    generator_layer_sizes=(32,64,32),
    critic_steps=5,
    n_list=(1,2,3,4),

    # --- optimisation --------------------------------------------------
    batch_size=32,
    ae_epochs=100,
    gan_epochs=100,
    feature_matching_weight=0.0,
    reconstruction_weight=0.0,
    generator_weight=1.0,

    # --- dirs ----------------------------------------------------------
    work_dir='runs'
):
    """
    End-to-end training.  Returns (autoencoder, generator, discriminator)
    """
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(device)
    R,C,T = input_shape

    #  ----- bookkeeping ------------------------------------------------
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out_dir = os.path.join(work_dir, stamp)
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(out_dir)

    #  ----- simulation params & helpers --------------------------------
    params = Params()
    assert (R,C,T) == (params.R, params.C, params.T)

    def generate_batches(bs):
        all_positions, all_counts, all_N = [], [], []
        for _ in range(bs):
            pos, cnt, N = generate_vertex_electron_params(n_list, params)
            all_positions.append(pos); all_counts.append(cnt); all_N.append(N)

        max_N = max(n_list)
        cond = np.zeros((bs, max_N, 4))
        for i,(p,c,N) in enumerate(zip(all_positions, all_counts, all_N)):
            cond[i,:N,:] = np.hstack([p, c[:,None]])

        pulses = []
        for pos,cnt in zip(all_positions, all_counts):
            pulse,_ = simulate_fixed_vertex_electron(pos, cnt, params,
                            seed=(int(time.time()*1e3) % 2**31))
            pulses.append(pulse)
        pulses = np.asarray(pulses)                        # (B,R,C,T)
        pulses = pulses.reshape(bs, R*C, T)                # flatten XY
        return torch.tensor(pulses, dtype=torch.float32), \
               torch.tensor(cond,   dtype=torch.float32)

    #  ----- models -----------------------------------------------------
    ae   = Autoencoder((R*C, T), latent_dim,
                       encoder_layer_sizes, decoder_layer_sizes).to(device)
    gen  = ConvDistributionGenerator((R,C,T), latent_dim,
                       layer_sizes=generator_layer_sizes).to('cpu')
    disc = Conv3DDiscriminator((R,C,T)).to(device)

    trainer = GANTrainer(ae, gen, disc,
                         reconstruction_weight=reconstruction_weight,
                         generator_weight=generator_weight,
                         feature_matching_weight=feature_matching_weight,
                         critic_steps=critic_steps)  # to(device) handled inside

    #  ----- optional autoencoder pre-training --------------------------
    if ae_epochs:
        print('Phase 1 – Autoencoder pre-training')
        optim_ae = torch.optim.Adam(ae.parameters(), 1e-4)
        mse = nn.MSELoss()
        for epoch in range(1, ae_epochs+1):
            ae.train()
            x,_ = generate_batches(batch_size)
            x = x.to(device)
            optim_ae.zero_grad()
            recon = ae(x)
            loss = mse(recon, x)
            loss.backward(); optim_ae.step()

            if epoch % 10 == 0:
                ae.eval()
                with torch.no_grad():
                    val,_ = generate_batches(batch_size)
                    val = val.to(device)
                    val_loss = mse(ae(val), val).item()
                print(f'AE [{epoch:>4}/{ae_epochs}]  loss={loss.item():.4f}  val={val_loss:.4f}')
                writer.add_scalar('ae_train_loss', loss.item(), epoch)
                writer.add_scalar('ae_val_loss',   val_loss,   epoch)

        # freeze encoder
        for p in ae.encoder.parameters():
            p.requires_grad_(False)

    #  ----- GAN training ----------------------------------------------
    print('Phase 2 – GAN')
    metrics_hist = {k:[] for k in
        ['generator_loss','critic_loss','wasserstein_distance',
         'gradient_penalty','reconstruction_loss',
         'critic_accuracy','generator_fool_rate']}

    for epoch in range(1, gan_epochs+1):
        t0 = time.time()
        real_x, cond = generate_batches(batch_size)
        stats = trainer.train_step(real_x, cond)

        for k,v in stats.items():
            if k in metrics_hist: metrics_hist[k].append(v)
            writer.add_scalar(f'gan/{k}', v, epoch+ae_epochs)

        if epoch % 100 == 0 or epoch == gan_epochs:
            torch.save(gen.state_dict(),
                       os.path.join(out_dir, f'gen_ep{epoch}.pt'))
            torch.save(disc.state_dict(),
                       os.path.join(out_dir, f'disc_ep{epoch}.pt'))

        if epoch % 1 == 0:
            dur = time.time()-t0
            print(f'GAN [{epoch:>5}/{gan_epochs}] '
                  f'G={stats["gen_loss"]:.3f} '
                  f'D={stats["disc_loss"]:.3f} '
                  f'W={stats["wasserstein"]:.3f}  ({dur:.1f}s)')

        if epoch % 20 == 0:
            generate_and_save_sample(ae, gen, real_x, epoch, params, epoch, sample_dir=out_dir)

    #  ----- final plots / samples -------------------------------------
    if gan_epochs:
        plot_training_metrics(metrics_hist,
            save_path=os.path.join(out_dir,'metrics.png'))

    return ae, gen, disc


# ======================================================================
# 2)  Quick test-run
# ======================================================================
if __name__ == '__main__':
    p = Params()
    ae, gen, disc = train_dense_gan(
        input_shape=(p.R, p.C, p.T),
        latent_dim=128,
        encoder_layer_sizes=(512,512),
        decoder_layer_sizes=(512,512),
        generator_layer_sizes=(256,512,512),
        critic_steps=2,
        batch_size=8,
        ae_epochs=0,
        gan_epochs=2000,
        feature_matching_weight=0.0,
        work_dir='runs_torch'
    )
