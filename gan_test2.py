# -------------------------------------------------------------------
#  Voxel-space DDPM (Score-based Diffusion) for 3-D Volumes
#  -- Stand-alone TensorFlow/Keras script (May 2025) --
# -------------------------------------------------------------------
#  This script mirrors the structure of the user-supplied WGAN-GP file
#  but implements a denoising diffusion probabilistic model (DDPM)
#  to generate noisy, high-frequency 32×32×32 voxel grids.
# -------------------------------------------------------------------
#  Author: ChatGPT (OpenAI o3)
# -------------------------------------------------------------------

import os, re, math, sys, time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# -------------------------------------------------------------------
#  Parameters (comparable to the original GAN script)
# -------------------------------------------------------------------
VOL_SHAPE   = (24, 24, 700)      # 3-D volume dimensions
VOL_DIM     = np.prod(VOL_SHAPE)
CHANNELS    = 1                 # single-channel voxel value
BATCH_SIZE  = 8
STEPS       = 10000            # optimisation steps (not diffusion steps!)
GIF_SAMPLES = 1                # how many volumes to visualise at checkpoints

# Diffusion hyper-parameters
T           = 1000             # number of diffusion time-steps
BETA_START  = 1e-4
BETA_END    = 0.02

LEARNING_RATE = 1e-4
EMA_DECAY     = 0.999          # EMA for the model used during sampling

# -------------------------------------------------------------------
#  Helper: cosine schedule (optional). We stick with linear β here for
#  simplicity but leave the cosine function as a ready alternative.
# -------------------------------------------------------------------

def linear_beta_schedule(timesteps, beta_start, beta_end):
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = np.arange(timesteps + 1, dtype=np.float64)
    alphas_cumprod = np.cos(((steps / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-7, 0.999)

# Choose schedule: linear or cosine
betas = linear_beta_schedule(T, BETA_START, BETA_END)
# betas = cosine_beta_schedule(T)  # uncomment to switch

alphas          = 1.0 - betas
alphas_cumprod  = np.cumprod(alphas, axis=0)

# Convert to tensors once for efficiency
betas_tf         = tf.constant(betas, dtype=tf.float32)
alphas_tf        = tf.constant(alphas, dtype=tf.float32)
alphas_bar_tf    = tf.constant(alphas_cumprod, dtype=tf.float32)

def plot_events(events, title='hit_pattern', subtitles=[]):
    assert len(events.shape) == 4, 'Events must be a 3D array with shape (num_events, num_rows * num_cols, num_samples)'
    events = np.transpose(events, axes=[3, 0, 1, 2])

    gif_frames = []
    for sample in tqdm(events):
        num_events = sample.shape[0]
        fig, ax = plt.subplots(1, num_events, figsize=(3*num_events, 3), dpi=100)
        fig.suptitle(title, fontsize=16)

        if num_events == 1: ax = [ax]

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
    gif_frames[0].save(
        filename,
        save_all = True,
        duration = 20,
        loop = 0,
        append_images = gif_frames[1:]
    )

    return filename

def make_dataset(batch_size=BATCH_SIZE):
    def generator():
        while True:
            yield np.random.normal(size=VOL_SHAPE + (CHANNELS,)).astype(np.float32)

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(shape=VOL_SHAPE + (CHANNELS,), dtype=tf.float32)
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

dataset = make_dataset()
iter_dataset = iter(dataset)

# -------------------------------------------------------------------
#  3-D U-Net backbone for ε-prediction
# -------------------------------------------------------------------

def conv_block(x, filters):
    x = tf.keras.layers.Conv3D(filters, 3, padding='same', activation='swish')(x)
    x = tf.keras.layers.Conv3D(filters, 3, padding='same', activation='swish')(x)
    return x

def make_unet():
    inputs = tf.keras.Input(shape=VOL_SHAPE + (CHANNELS,), name='noised_volume')

    # Encoding
    c1 = conv_block(inputs, 64)
    p1 = tf.keras.layers.AveragePooling3D(pool_size=2)(c1)

    c2 = conv_block(p1, 128)
    p2 = tf.keras.layers.AveragePooling3D(pool_size=2)(c2)

    # Bottleneck
    bn = conv_block(p2, 256)

    # Decoding
    u2 = tf.keras.layers.Conv3DTranspose(128, 3, strides=2, padding='same')(bn)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    c3 = conv_block(u2, 128)

    u1 = tf.keras.layers.Conv3DTranspose(64, 3, strides=2, padding='same')(c3)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    c4 = conv_block(u1, 64)

    outputs = tf.keras.layers.Conv3D(CHANNELS, 1, padding='same', activation='linear')(c4)
    return tf.keras.Model(inputs, outputs, name='voxel_unet_eps')

model = make_unet()
ema_model = tf.keras.models.clone_model(model)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# -------------------------------------------------------------------
#  Training step (DDPM loss = MSE between ε and ε̂)
# -------------------------------------------------------------------
@tf.function
def train_step(x0):
    batch_size = tf.shape(x0)[0]
    t = tf.random.uniform(minval=0, maxval=T, shape=(batch_size,), dtype=tf.int32)

    alphas_bar = tf.gather(alphas_bar_tf, t)
    alphas_bar = tf.reshape(alphas_bar, (-1, 1, 1, 1, 1))

    noise = tf.random.normal(tf.shape(x0))
    xt = tf.sqrt(alphas_bar) * x0 + tf.sqrt(1.0 - alphas_bar) * noise

    with tf.GradientTape() as tape:
        noise_pred = model(xt, training=True)
        loss = tf.reduce_mean((noise - noise_pred) ** 2)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # EMA update
    for ew, mw in zip(ema_model.trainable_variables, model.trainable_variables):
        ew.assign(EMA_DECAY * ew + (1 - EMA_DECAY) * mw)

    return loss

# -------------------------------------------------------------------
#  Sampling (DDPM ancestral sampler a-la Ho et al., 2020)
# -------------------------------------------------------------------

def sample_ddpm(num_samples=GIF_SAMPLES, use_ema=True):
    model_to_use = ema_model if use_ema else model
    xt = tf.random.normal((num_samples,) + VOL_SHAPE + (CHANNELS,))

    for t in reversed(range(T)):
        beta_t = betas_tf[t]
        alpha_t = alphas_tf[t]
        alpha_bar_t = alphas_bar_tf[t]

        eps = model_to_use(xt, training=False)

        coef1 = 1 / tf.sqrt(alpha_t)
        coef2 = beta_t / tf.sqrt(1 - alpha_bar_t)
        x0_pred = coef1 * (xt - coef2 * eps)

        if t > 0:
            noise = tf.random.normal(tf.shape(xt))
            beta_tilde = (1 - alpha_bar_t) / (1 - alphas_bar_tf[t-1]) * beta_t
            xt = tf.sqrt(alphas_tf[t-1]) * x0_pred + tf.sqrt(beta_tilde) * noise
        else:
            xt = x0_pred
    return xt.numpy()

# -------------------------------------------------------------------
#  Training loop with simple logging and visual checkpoints
# -------------------------------------------------------------------
losses = []
ema_losses = []
ema_alpha = 0.9

print("Starting DDPM training ...")
for step in range(1, STEPS + 1):
    x0_batch = next(iter_dataset)
    loss = train_step(x0_batch)
    loss_val = loss.numpy()
    losses.append(loss_val)

    # EMA of loss for smoother plot
    if step == 1:
        ema_losses.append(loss_val)
    else:
        ema_losses.append(ema_alpha * ema_losses[-1] + (1 - ema_alpha) * loss_val)

    print(f"Step {step:05d}  Loss = {loss_val:.4f}")

    # Visual checkpoint every 1000 steps
    if step % 100 == 1 or step == STEPS:
        samples = sample_ddpm(num_samples=GIF_SAMPLES)
        true_samples = x0_batch
        samples_concat = np.concatenate([true_samples[0:1], samples[0:1]], axis=0)
        samples_concat_squeezed = np.squeeze(samples_concat)
        print(samples_concat_squeezed.shape)
        gif_file = plot_events(samples_concat_squeezed, title=f'ddpm_step_{step}', subtitles=[f'Sample {i}' for i in range(GIF_SAMPLES)])
        print(f"Saved samples to {gif_file}")

print("Training complete!")

# -------------------------------------------------------------------
#  Plot training curve
# -------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(range(1, STEPS + 1), losses, 'b-', alpha=0.3, label='Loss')
plt.plot(range(1, STEPS + 1), ema_losses, 'b-', linewidth=2, label='Loss (EMA)')
plt.xlabel('Steps')
plt.ylabel('MSE')
plt.title('DDPM Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('ddpm_training_plot.png', dpi=300)
plt.close()
print("Training plot saved to ddpm_training_plot.png")

# -------------------------------------------------------------------
#  Usage:
#      ✦ Replace make_dataset() with your real volume loader.
#      ✦ Tweak VOL_SHAPE / network depth for higher resolution.
#      ✦ For faster sampling, integrate DPM-Solver++ or DDIM.
# -------------------------------------------------------------------
