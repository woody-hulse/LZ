import os, re, math, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# -------------------------------------------------------------------
#  Parameters
# -------------------------------------------------------------------
VOL_SHAPE  = (32, 32, 32)          # 3‑D volume dimensions
VOL_DIM    = np.prod(VOL_SHAPE)    # flattened length (32768)
LATENT_DIM = 64
BATCH_SIZE = 8
STEPS      = 10000
CRITIC_STEPS = 5
GP_WEIGHT  = 10.0
GIF_SAMPLES= 8                     # how many volumes to visualise

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
            yield np.random.normal(0, 1, size=(VOL_DIM,)).reshape(-1)  # flatten to 1‑D
    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(shape=(VOL_DIM,), dtype=tf.float32)
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

dataset = make_dataset()
ds_iter = iter(dataset)

def make_generator():
    inputs = tf.keras.Input(shape=(LATENT_DIM,))
    x = tf.keras.layers.Dense(512, activation="sigmoid")(inputs)
    x = tf.keras.layers.Dense(1024, activation="sigmoid")(x)
    x = tf.keras.layers.Dense(2048, activation="sigmoid")(x)
    outputs = tf.keras.layers.Dense(VOL_DIM, activation="linear")(x)
    return tf.keras.Model(inputs, outputs, name="dense_generator")

def make_distribution_generator():
    inputs = tf.keras.Input(shape=(LATENT_DIM,))
    x = tf.keras.layers.Dense(512, activation="sigmoid")(inputs)
    x = tf.keras.layers.Dense(1024, activation="sigmoid")(x)
    x = tf.keras.layers.Dense(2048, activation="sigmoid")(x)
    mu_outputs = tf.keras.layers.Dense(VOL_DIM, activation="linear")(x)
    sigma_outputs = tf.keras.layers.Dense(VOL_DIM, activation="softplus")(x)
    

    return tf.keras.Model(inputs, [mu_outputs, sigma_outputs], name="distribution_generator")

class DistributionGenerator(tf.keras.Model):
    def __init__(self, layer_sizes=[512, 1024, 2048], **kwargs):
        super(DistributionGenerator, self).__init__(**kwargs)
        self.layers_list = []
        for units in layer_sizes:
            self.layers_list.append(tf.keras.layers.Dense(units, activation="sigmoid"))
        self.mu_output = tf.keras.layers.Dense(VOL_DIM, activation="linear")
        self.sigma_output = tf.keras.layers.Dense(VOL_DIM, activation="softplus")

    def call(self, inputs):
        for layer in self.layers_list:
            inputs = layer(inputs)
        mu = self.mu_output(inputs)
        sigma = self.sigma_output(inputs)
        output = mu + sigma * tf.random.normal(tf.shape(mu))
        return output

def make_generator_with_noise_injection():
    """
    Generator with per‑layer noise‑injection:
      for each hidden layer, concat(x, ϵ)  →  Dense → relu
    where ϵ ~ N(0,1) and has the same shape as x.
    """
    def inject_noise(t, a=2):
        for i in range(a):
            noise = tf.random.normal(tf.shape(t))
            t = tf.concat([t, noise], axis=-1)
        return t

    inputs = tf.keras.Input(shape=(LATENT_DIM,))

    x = tf.keras.layers.Lambda(inject_noise)(inputs)
    for units in (128, 128, 128):
        # x = tf.keras.layers.Lambda(inject_noise)(x)
        x = tf.keras.layers.Dense(units, activation='sigmoid')(x)

    outputs = tf.keras.layers.Dense(VOL_DIM, activation="linear")(x)
    return tf.keras.Model(inputs, outputs, name="dense_generator_with_noise_injection")

def make_discriminator():
    inputs = tf.keras.Input(shape=(VOL_DIM,))
    x = tf.keras.layers.Dense(1024, activation="relu")(inputs)
    x = tf.keras.layers.Dense(512,  activation="relu")(x)
    x = tf.keras.layers.Dense(256,  activation="relu")(x)
    
    # Add minibatch discrimination features
    batch_features = MinibatchDiscrimination(32, 16)(x)
    x = tf.keras.layers.Concatenate()([x, batch_features])
    
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)  # WGAN score
    return tf.keras.Model(inputs, outputs, name="dense_critic")

class MinibatchDiscrimination(tf.keras.layers.Layer):
    def __init__(self, num_kernels, kernel_dim, **kwargs):
        super(MinibatchDiscrimination, self).__init__(**kwargs)
        self.num_kernels = num_kernels
        self.kernel_dim = kernel_dim
        
    def build(self, input_shape):
        self.T = self.add_weight(
            name='minibatch_discrimination_weights',
            shape=(input_shape[-1], self.num_kernels * self.kernel_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        super(MinibatchDiscrimination, self).build(input_shape)
    
    def call(self, inputs):
        # Transform inputs into M: batch_size x num_kernels x kernel_dim
        M = tf.matmul(inputs, self.T)
        M = tf.reshape(M, [-1, self.num_kernels, self.kernel_dim])
        
        # Expand and compute L1 distance between points in minibatch
        # Expand to shape (batch_size, 1, num_kernels, kernel_dim)
        M_expanded_a = tf.expand_dims(M, 1)
        # Expand to shape (1, batch_size, num_kernels, kernel_dim)
        M_expanded_b = tf.expand_dims(M, 0)
        # L1 distance, shape (batch_size, batch_size, num_kernels)
        L1_dist = tf.reduce_sum(tf.abs(M_expanded_a - M_expanded_b), axis=3)
        # Apply negative exponential to distance
        K = tf.exp(-L1_dist)
        # Sum across second batch dimension
        K_sum = tf.reduce_sum(K, axis=1) - 1.0  # Subtract 1 to exclude self-comparison
        return K_sum
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_kernels)
    
    def get_config(self):
        config = super(MinibatchDiscrimination, self).get_config()
        config.update({
            'num_kernels': self.num_kernels,
            'kernel_dim': self.kernel_dim
        })
        return config

G = DistributionGenerator() # make_generator()
D = make_discriminator()

g_opt = tf.keras.optimizers.Adam(5e-5)
d_opt = tf.keras.optimizers.Adam(5e-5)

@tf.function
def gradient_penalty(real, fake):
    eps = tf.random.uniform([real.shape[0], 1], 0.0, 1.0)
    inter = real + eps * (fake - real)
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(inter)
        score = D(inter, training=True)
    grads = gp_tape.gradient(score, inter)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
    return tf.reduce_mean((norm - 1.)**2)

@tf.function
def train_d(real):
    z = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    with tf.GradientTape() as tape:
        fake = G(z, training=True)
        d_real = D(real, training=True)
        d_fake = D(fake, training=True)
        gp = gradient_penalty(real, fake)
        loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + GP_WEIGHT*gp
    grads = tape.gradient(loss, D.trainable_variables)
    d_opt.apply_gradients(zip(grads, D.trainable_variables))
    return loss, d_real, d_fake

@tf.function
def train_g():
    z = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    with tf.GradientTape() as tape:
        fake = G(z, training=True)
        loss = -tf.reduce_mean(D(fake, training=True))
    grads = tape.gradient(loss, G.trainable_variables)
    g_opt.apply_gradients(zip(grads, G.trainable_variables))
    return loss

# Initialize lists to store losses for plotting
d_losses = []
g_losses = []
d_ema = []
g_ema = []
d_accuracies = []
d_acc_ema = []
ema_alpha = 0.9

for step in range(1, STEPS+1):
    # ----- critic steps -----
    for _ in range(CRITIC_STEPS):
        real_batch = next(ds_iter)
        d_loss, d_real, d_fake = train_d(real_batch)
    # ----- generator step -----
    g_loss = train_g()
    
    # Store losses
    d_loss_val = d_loss.numpy()
    g_loss_val = g_loss.numpy()
    d_losses.append(d_loss_val)
    g_losses.append(g_loss_val)
    
    # Calculate discriminator accuracy - % of time real scores higher than fake
    z = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    fake_samples = G(z, training=False)
    real_samples = next(ds_iter)
    
    d_real_eval = D(real_samples, training=False)
    d_fake_eval = D(fake_samples, training=False)
    
    # Compare each real sample with each fake sample
    real_expanded = tf.expand_dims(d_real_eval, 1)  # Shape: [batch, 1, 1]
    fake_expanded = tf.expand_dims(d_fake_eval, 0)  # Shape: [1, batch, 1]
    
    # Calculate how often real > fake (should be 100% for perfect discriminator)
    comparisons = tf.cast(real_expanded > fake_expanded, tf.float32)
    accuracy = tf.reduce_mean(comparisons).numpy()
    d_accuracies.append(accuracy)
    
    # Calculate exponential moving averages
    if step == 1:
        d_ema.append(d_loss_val)
        g_ema.append(g_loss_val)
        d_acc_ema.append(accuracy)
    else:
        d_ema.append(ema_alpha * d_ema[-1] + (1 - ema_alpha) * d_loss_val)
        g_ema.append(ema_alpha * g_ema[-1] + (1 - ema_alpha) * g_loss_val)
        d_acc_ema.append(ema_alpha * d_acc_ema[-1] + (1 - ema_alpha) * accuracy)

    print(f"Step {step:04d}  D_loss={d_loss_val:8.3f}  G_loss={g_loss_val:8.3f}  D_acc={accuracy:8.3f}")

    if step % 500 == 0:
        z = tf.random.normal([GIF_SAMPLES, LATENT_DIM])
        generated = (G(z, training=False).numpy()[0]).reshape(1, *VOL_SHAPE)
        real = (next(ds_iter).numpy()[0]).reshape(1, *VOL_SHAPE)
        concat = np.concatenate([real, generated], axis=0)

        gif_file = plot_events(concat, title=f"{G.name}_samples_{step}", subtitles=["Real", "Generated"])
        print(f"Saved GIF to {gif_file}")

print("Training done!")

# Plot training losses with EMA
plt.figure(figsize=(10, 6))
steps = range(1, STEPS+1)
plt.plot(steps, d_losses, 'b-', alpha=0.3, label='Critic Loss')
plt.plot(steps, g_losses, 'r-', alpha=0.3, label='Generator Loss')
plt.plot(steps, d_ema, 'b-', linewidth=2, label='Critic Loss (EMA)')
plt.plot(steps, g_ema, 'r-', linewidth=2, label='Generator Loss (EMA)')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('WGAN-GP Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('wgan_training_plot.png', dpi=300)
plt.close()
print("Training plot saved to wgan_training_plot.png")

# Plot discriminator accuracy
plt.figure(figsize=(10, 6))
plt.plot(steps, d_accuracies, 'g-', alpha=0.3, label='Discriminator Accuracy')
plt.plot(steps, d_acc_ema, 'g-', linewidth=2, label='Discriminator Accuracy (EMA)')
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Guess (0.5)')
plt.xlabel('Steps')
plt.ylabel('Accuracy (% real > fake)')
plt.title('Discriminator Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.savefig('discriminator_accuracy_plot.png', dpi=300)
plt.close()
print("Discriminator accuracy plot saved to discriminator_accuracy_plot.png")
