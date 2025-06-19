# -------------------------------------------------------------------
#  Residual‑only DDPM (RDDM flavour) for 3‑D TPC voxel volumes
#  Replaces the former WGAN‑GP with a 3‑D U‑Net noise‑predictor.
#  All support utilities (dataset, plotting, etc.) are retained.
# -------------------------------------------------------------------
import os, re, math, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# -------------------------------------------------------------------
#  Parameters (kept identical where possible)
# -------------------------------------------------------------------
VOL_SHAPE   = (32, 32, 32)           # 3‑D volume dimensions
VOL_DIM     = np.prod(VOL_SHAPE)     # flattened length (32768)
BATCH_SIZE  = 8
STEPS       = 1000                  # optimisation iterations
GIF_SAMPLES = 1                     # how many volumes to visualise

# DDPM / RDDM schedule
T_STEPS     = 4000                   # diffusion timesteps
BETA_START  = 1e-4
BETA_END    = 2e-2

# -------------------------------------------------------------------
#  Helper: cosine or linear beta schedule  (simple linear here)
# -------------------------------------------------------------------
betas = np.linspace(BETA_START, BETA_END, T_STEPS, dtype=np.float32)
alphas = 1.0 - betas
alpha_cumprod = np.cumprod(alphas, axis=0)
alpha_cumprod_prev = np.append(1.0, alpha_cumprod[:-1])

sqrt_alphas_cumprod        = np.sqrt(alpha_cumprod)
sqrt_one_minus_alpha_cum   = np.sqrt(1.0 - alpha_cumprod)

# -------------------------------------------------------------------
#  Utility:  FFT‑based PSD sanity check  (optional – can be commented)
# -------------------------------------------------------------------
def power_spectral_density(volumes):
    # Compute 1‑D FFT along drift‑time dimension (axis=2) and average PSD
    fft = tf.signal.rfft(volumes, axis=2)
    psd = tf.reduce_mean(tf.abs(fft) ** 2, axis=[0, 1, 3])
    return psd

# -------------------------------------------------------------------
#  Plotting utility from original script (unchanged)
# -------------------------------------------------------------------

def plot_events(events, title='hit_pattern', subtitles=[]):
    assert len(events.shape) == 4, 'Events must be (num, x, y, z)'
    events = np.transpose(events, axes=[3, 0, 1, 2])  # depth axis first for GIF scroll

    gif_frames = []
    for sample in tqdm(events):
        num_events = sample.shape[0]
        fig, ax = plt.subplots(1, num_events, figsize=(3*num_events, 3), dpi=100)
        fig.suptitle(title, fontsize=16)

        if num_events == 1:
            ax = [ax]

        for i, hit_pattern in enumerate(sample):
            ax[i].imshow(hit_pattern, vmin=0, vmax=5)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].grid(False)
            if len(subtitles) >= i + 1:
                ax[i].set_title(subtitles[i], fontsize=8)

        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.dpi
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image_array = data.reshape(int(height), int(width), 4)

        image_frame = Image.fromarray(image_array)
        gif_frames.append(image_frame)
        plt.clf()
        plt.close()

    filename = re.sub(r'[^a-zA-Z0-9]', '', title.lower()) + '_gif.gif'
    gif_frames[0].save(filename, save_all=True, duration=20, loop=0, append_images=gif_frames[1:])
    return filename

# -------------------------------------------------------------------
#  Dataset – keep identical dummy generator (replace with residuals)
# -------------------------------------------------------------------

def make_dataset(batch_size=BATCH_SIZE):
    def generator():
        while True:
            # Dummy residuals for example: standard Gaussian noise in each voxel
            yield np.random.normal(0, 1, size=(VOL_DIM,)).astype(np.float32)
    ds = tf.data.Dataset.from_generator(generator, output_signature=tf.TensorSpec(shape=(VOL_DIM,), dtype=tf.float32))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

dataset = make_dataset()
ds_iter = iter(dataset)

# -------------------------------------------------------------------
#  3‑D U‑Net noise‑predictor  ε_θ(x_t, t)
# -------------------------------------------------------------------

def down_block(x, filters):
    x = tf.keras.layers.Conv3D(filters, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv3D(filters, 3, padding='same', activation='relu')(x)
    skip = x
    x = tf.keras.layers.MaxPool3D()(x)
    return x, skip

def up_block(x, skip, filters):
    x = tf.keras.layers.Conv3DTranspose(filters, 2, strides=2, padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = tf.keras.layers.Conv3D(filters, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv3D(filters, 3, padding='same', activation='relu')(x)
    return x


def make_unet_eps_model(vol_shape=VOL_SHAPE):
    t_emb_dim = 128  # dimensionality of sinusoidal timestep embedding

    # --- inputs
    vol_in = tf.keras.Input(shape=vol_shape + (1,), name='noisy_volume')
    t_in   = tf.keras.Input(shape=(), dtype=tf.int32, name='diffusion_step')

    # --- timestep embedding (sinusoidal)  -> dense -> dense + GELU
    def timestep_embedding(t, dim: int = t_emb_dim):
        half = dim // 2
        emb = tf.cast(t, tf.float32)[:, None]  # (B,1)
        freq = tf.exp(tf.range(half, dtype=tf.float32) * -(math.log(10000.0) / half))
        emb = emb * freq
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb

    t_emb = tf.keras.layers.Lambda(lambda s: timestep_embedding(s))(t_in)
    t_emb = tf.keras.layers.Dense(t_emb_dim, activation='gelu')(t_emb)
    t_emb = tf.keras.layers.Dense(t_emb_dim, activation='gelu')(t_emb)

    # --- Down path
    x = vol_in
    x_shape = x.shape[1:]
    skips = []
    for f in (32, 64, 128):
        x, s = down_block(x, f)
        skips.append(s)

    # --- Bottleneck
    x = tf.keras.layers.Conv3D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv3D(256, 3, padding='same', activation='relu')(x)

    # inject timestep embedding at bottleneck
    bottleneck_shape = x.shape[1:4]
    t_proj = tf.keras.layers.Dense(bottleneck_shape[0] * bottleneck_shape[1] * bottleneck_shape[2] * 256, activation='gelu')(t_emb)
    t_proj = tf.keras.layers.Reshape((bottleneck_shape[0], bottleneck_shape[1], bottleneck_shape[2], 256))(t_proj)
    x = tf.keras.layers.Add()([x, t_proj])

    # --- Up path
    for f, skip in zip((128, 64, 32)[::-1], skips[::-1]):
        x = up_block(x, skip, f)

    # --- Output layer predicts ε (same shape as input)
    out = tf.keras.layers.Conv3D(1, 1, padding='same', activation=None)(x)

    return tf.keras.Model([vol_in, t_in], out, name='unet_eps_predictor')

model = make_unet_eps_model()
model.summary()

optimizer = tf.keras.optimizers.Adam(1e-4)

# -------------------------------------------------------------------
#  Training step (DDPM‑style): predict ε and minimise MSE
# -------------------------------------------------------------------
@tf.function
def train_step(x0_flat):
    # reshape flat -> (B, D, H, W, 1)
    x0 = tf.reshape(x0_flat, [-1, *VOL_SHAPE, 1])
    bsz = tf.shape(x0)[0]

    # sample random timestep t ∈ [1, T]
    t = tf.random.uniform([bsz], minval=0, maxval=T_STEPS, dtype=tf.int32)
    a_bar = tf.gather(sqrt_alphas_cumprod, t)
    sigma = tf.gather(sqrt_one_minus_alpha_cum, t)

    # sample ε ~ N(0,1)
    eps = tf.random.normal(tf.shape(x0))

    # forward diffusion: x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε
    a_bar = tf.reshape(a_bar, [-1, 1, 1, 1, 1])
    sigma = tf.reshape(sigma, [-1, 1, 1, 1, 1])
    xt = a_bar * x0 + sigma * eps

    with tf.GradientTape() as tape:
        eps_pred = model([xt, t], training=True)
        loss = tf.reduce_mean((eps - eps_pred)**2)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# -------------------------------------------------------------------
#  Sampling (DDPM ancestral sampler, can be replaced with DDIM)
# -------------------------------------------------------------------
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, *VOL_SHAPE, 1], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.int32)
])
def p_sample(x, t):
    t_int = tf.cast(t, tf.int32)
    beta_t = tf.gather(betas, t_int)
    sqrt_one_minus_alpha_cum_t = tf.gather(sqrt_one_minus_alpha_cum, t_int)
    sqrt_alpha_t = tf.sqrt(1. - beta_t)

    eps_theta = model([x, tf.fill([tf.shape(x)[0]], t_int)])

    # x_{t-1}
    coef = 1.0 / tf.sqrt(1.0 - beta_t)
    mean = coef * (x - beta_t / sqrt_one_minus_alpha_cum_t * eps_theta)

    # Conditional based on tensor value, not Python value
    should_add_noise = tf.greater(t_int, 0)
    noise = tf.random.normal(tf.shape(x))
    var = tf.sqrt(beta_t) * noise
    # Use where instead of if-else
    return tf.cond(should_add_noise, lambda: mean + var, lambda: mean)


def sample(n_samples=1):
    x = tf.random.normal([n_samples, *VOL_SHAPE, 1])
    for t in tqdm(reversed(range(T_STEPS))):
        x = p_sample(x, tf.constant(t))
    x = tf.reshape(x, [n_samples, *VOL_SHAPE])
    return x

# -------------------------------------------------------------------
#  Training loop
# -------------------------------------------------------------------
loss_log = []
ema_loss_log = []
ema_alpha = 0.9

for step in range(1, STEPS + 1):
    real_batch = next(ds_iter)
    loss = train_step(real_batch)
    loss_val = loss.numpy()
    loss_log.append(loss_val)
    if step == 1:
        ema_loss_log.append(loss_val)
    else:
        ema_loss_log.append(ema_alpha * ema_loss_log[-1] + (1 - ema_alpha) * loss_val)

    print(f"Step {step:05d}  Loss={loss_val:8.5f}")

    # visualisation
    if step % 100 == 0 or step == STEPS:
        gen_vols = sample(GIF_SAMPLES).numpy()
        real_vols = tf.reshape(real_batch[:GIF_SAMPLES], [GIF_SAMPLES, *VOL_SHAPE]).numpy()
        concat = np.concatenate([real_vols, gen_vols], axis=0)
        gif_file = plot_events(concat, title=f"rddm_samples_{step}", subtitles=["Real"]*GIF_SAMPLES + ["Generated"]*GIF_SAMPLES)
        print(f"Saved GIF to {gif_file}")

print("Training done!")

# -------------------------------------------------------------------
#  Plot training loss
# -------------------------------------------------------------------
plt.figure(figsize=(10, 6))
steps_axis = range(1, len(loss_log) + 1)
plt.plot(steps_axis, loss_log, 'b-', alpha=0.3, label='MSE Loss')
plt.plot(steps_axis, ema_loss_log, 'b-', linewidth=2, label='Loss (EMA)')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('RDDM Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('rddm_training_plot.png', dpi=300)
plt.close()
print("Training plot saved to rddm_training_plot.png")