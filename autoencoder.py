import tensorflow as tf
import tensorflow.keras.backend as K
import spektral
from spektral.utils import normalized_laplacian, rescale_laplacian

import re
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap

from preprocessing import *
from pulse import *
from generator import *

from regression_models import GATMultivariateNormalModel, DenseMultivariateNormalModel
from regression_models import GATNumScattersModel, DenseNumScattersModel

def set_mpl_style():
    plt.rcParams.update({
        "figure.dpi": 300,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "lines.markersize": 2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": True,
        "axes.grid": False,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
    })


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


def vis_latent_space_categories(data_categories, data_categories_labels, title):
    # plt.rcParams['figure.dpi'] = 120
    
    for data, label in zip(data_categories, data_categories_labels):
        plt.scatter(data[:, 0], data[:, 1], label=label, s=0.5)
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.legend()
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(re.sub(r'[^a-zA-Z0-9]', '', title.lower()) + '.png')
    # plt.show()
    plt.clf()
    
def vis_latent_space_gradients(latent_space, labels, title, colorbar_label):
    # plt.rcParams['figure.dpi'] = 120
    
    plt.scatter(latent_space[:, 0], latent_space[:, 1], c=labels, cmap='viridis', s=0.5)
    plt.title(title)
    plt.colorbar().set_label(colorbar_label)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(re.sub(r'[^a-zA-Z0-9]', '', title.lower()) + '.png')
    # plt.show()
    plt.clf()
    

def vis_latent_space_num_scatters(fit_model, latent_space, Y, N=4):
    num_scatters = np.where(Y > 0, 1, 0).sum(axis=(1, 2))
    
    num_sactters_categories = [latent_space[np.where(num_scatters == i)] for i in range(1, N + 1)]
    num_scatters_labels = [f'{i} scatter' + ('s' if i > 1 else '') for i in range(1, N + 1)]
    
    vis_latent_space_categories(num_sactters_categories, num_scatters_labels, f'Autoencoder Latent Space by Number of Scatters {type(fit_model).__name__.upper()}')


def vis_latent_space_phd(fit_model, latent_space, XC):
    phd = XC.sum(axis=(1, 2))
    vis_latent_space_gradients(latent_space, phd, f'Autoencoder Latent Space by Total Photoelectrons Deposited {type(fit_model).__name__.upper()}', colorbar_label='phd')
    
def vis_latent_space_footprint(fit_model, latent_space, XC):
    footprint = np.where(XC.sum(axis=-1) > 4, 1, 0).sum(axis=-1)
    
    vis_latent_space_gradients(latent_space, footprint, f'Autoencoder Latent Space by Footprint Size {type(fit_model).__name__.upper()}', colorbar_label='Total # PMT')
    

def codebook_usage_histogram(vqvae, XC):
    indices = vqvae.encode_to_indices_probabilistic(XC).numpy().flatten().astype(int)
    codebook_usage = np.bincount(indices, minlength=vqvae.num_embeddings)
    codebook_usage_sorted = codebook_usage[np.argsort(codebook_usage)[::-1]] / codebook_usage.sum()

    # plt.rcParams['figure.dpi'] = 120
    
    plt.fill_between(np.arange(len(codebook_usage_sorted)), codebook_usage_sorted, color='blue', alpha=0.3)
    plt.plot(np.arange(len(codebook_usage_sorted)), codebook_usage_sorted, color='blue', label='Usage')

    plt.xlabel('Codebook Index')
    plt.ylabel('Usage (PDF)')
    plt.title('VQ-VAE Codebook Usage Distribution')
    plt.margins(0)
    plt.savefig('codebook_usage.png')
    # plt.show()
    plt.clf()


def get_encoded_data_generator(compression_func, data_generator):
    while True:
        XC, XYZ, P = next(iter(data_generator))
        XC_encoded = compression_func(XC)
        yield XC_encoded, XYZ, P

def run_aux_task(models, compression_funcs, labels, data_generator, val_data_generator, fname_suffix=''):
    def precompute_batches(generator, steps):
        return [next(generator) for _ in tqdm(range(steps))]

    def train_epoch(model, train_batches, desc=''):
        total_loss = 0.0
        for batch in tqdm(train_batches, desc=desc, ncols=100):
            x, y, _ = batch
            loss = model.step(x, y, training=True)
            if isinstance(loss, (list, tuple)):
                loss = loss[0]
            total_loss += loss
        return total_loss

    def test_epoch(model, val_batches):
        total_loss = 0.0
        for batch in val_batches:
            x, y, _ = batch
            loss = model.step(x, y, training=False)
            if isinstance(loss, (list, tuple)):
                loss = loss[0]
            total_loss += loss
        return total_loss
    
    def train_and_plot_model(model, compression_func, data_generator, val_data_generator, epochs=10, steps_per_epoch=64, val_steps=4):
        if compression_func is not None:
            data_generator = get_encoded_data_generator(compression_func, data_generator)
            val_data_generator = get_encoded_data_generator(compression_func, val_data_generator)
        
        initial_val_loss = test_epoch(model, precompute_batches(val_data_generator, val_steps)) / val_steps
        
        train_times, val_losses = [0], [initial_val_loss]
        
        for epoch in range(epochs):
            train_batches = precompute_batches(data_generator, steps_per_epoch)
            val_batches = precompute_batches(data_generator, val_steps)
            
            desc = f'epoch {epoch + 1}/{epochs} ({model.name})'
            t0 = time.time()
            loss = train_epoch(model, train_batches, desc=desc)
            train_time = time.time() - t0
            avg_loss = loss / steps_per_epoch
            avg_val_loss = test_epoch(model, val_batches) / val_steps
            print(desc + f' - loss: {avg_loss:.3f}, val_loss: {avg_val_loss:.3f}')
            
            train_times.append(train_time)
            val_losses.append(avg_val_loss)
                
        K.clear_session()
            
        cdf_train_times = [sum(train_times[:i + 1]) for i in range(len(train_times))]
        best_val_losses = [min(val_losses[:i + 1]) for i in range(len(val_losses))]
            
        return cdf_train_times, best_val_losses
    
    epochs = 50
    steps_per_epoch = 64
    val_steps = 8
            
    model_cdf_train_times, model_best_val_losses = [], []
    
    for model, compression_func in zip(models, compression_funcs):
        cdf_train_times, best_val_losses = train_and_plot_model(model, compression_func, data_generator, val_data_generator, epochs=epochs, steps_per_epoch=steps_per_epoch, val_steps=val_steps)
        model_cdf_train_times.append(cdf_train_times)
        model_best_val_losses.append(best_val_losses)
    
    sample_batch, _, _ = next(iter(data_generator))
    batch_size = sample_batch.shape[0]
    
    plt.figure()
    epochs_axis = np.arange(0, epochs + 1) * steps_per_epoch * batch_size
    for label, val_losses in zip(labels, model_best_val_losses):
        plt.plot(epochs_axis, val_losses, '-o', label=label, markersize=2)
    plt.xlabel('Training Samples')
    plt.ylabel('Validation Loss')
    plt.title('Sample Efficiency: Validation Loss (Best) vs Training Samples')
    plt.legend()
    plt.savefig(f'raw_vs_compressed_sample_efficiency{fname_suffix}.png')
    
    plt.figure()
    for label, cdf_train_times, val_losses in zip(labels, model_cdf_train_times, model_best_val_losses):
        plt.plot(cdf_train_times, val_losses, '-o', label=label, markersize=2)
    plt.xlabel('Training Time (s)')
    plt.ylabel('Validation Loss')
    plt.title('Time Efficiency: Validation Loss (Best) vs Training Time')
    plt.legend()
    plt.savefig(f'raw_vs_compressed_time_efficiency{fname_suffix}.png')

class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, ckpt_manager):
        super().__init__()
        self.ckpt_manager = ckpt_manager

    def on_epoch_end(self, epoch, logs=None):
        save_path = self.ckpt_manager.save()
        print(f'\nsaved checkpoint for epoch {epoch + 1}: {save_path}')


@tf.keras.utils.register_keras_serializable(package='Custom', name='Encoder')
class Encoder(tf.keras.Model):
    def __init__(self, name='encoder'):
        super(Encoder, self).__init__(name=name)
    
    def call(self, x):
        pass
    
    def encode(self, x):
        pass
    
    def decode(self, x):
        pass
    
    def compress(self, x):
        pass
    
    def get_data_size_reduction(self):
        pass
        

@tf.keras.utils.register_keras_serializable(package='Custom', name='Autoencoder')
class Autoencoder(Encoder):
    def __init__(self, input_shape, latent_dim, encoder_layer_sizes=[], decoder_layer_sizes=[], name='autoencoder'):
        super(Autoencoder, self).__init__(name=name)
        
        self.input_shape = input_shape
        self.encoder_layer_sizes = encoder_layer_sizes
        self.decoder_layer_sizes = decoder_layer_sizes
        self.latent_dim = latent_dim
        
        self.encoder = tf.keras.Sequential(
            [tf.keras.layers.Input(input_shape)] +
            [tf.keras.layers.Flatten()] +
            [tf.keras.layers.Dense(sz, activation='relu') for sz in encoder_layer_sizes] +
            [tf.keras.layers.Dense(latent_dim)], name=f'{name}_encoder'
        )
    
        self.decoder = tf.keras.Sequential(
            [tf.keras.layers.Input((latent_dim,))] +
            [tf.keras.layers.Dense(sz, activation='relu') for sz in decoder_layer_sizes] +
            [tf.keras.layers.Dense(np.prod(input_shape), activation='softplus'),
             tf.keras.layers.Reshape(input_shape)], name=f'{name}_decoder'
        )
    
    def call(self, x):
        return self.decoder(self.encoder(x))
    
    def encode(self, x):
        return self.encoder(x)
    
    def compress(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def compile(self, optimizer, loss, metrics=[]):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.optimizer = optimizer
        self.loss = loss
        
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
    
    def from_config(self, config):
        return Autoencoder(**config)
    
    def get_config(self):
        return {
            'input_shape': self.input_shape,
            'latent_dim': self.latent_dim,
            'encoder_layer_sizes': self.encoder_layer_sizes,
            'decoder_layer_sizes': self.decoder_layer_sizes
        }
        
    def build(self, input_shape):
        self.input_shape = input_shape
        super().build(input_shape)
        
    def get_data_size_reducton(self):
        return np.prod(self.input_shape) / self.latent_dim
        

@tf.keras.utils.register_keras_serializable(package='Custom', name='AutoencoderLoss')
def reconstruction_loss(x, reconstructed, **kwargs):
    return tf.reduce_mean(tf.keras.losses.MeanSquaredError()(x, reconstructed))


@tf.keras.utils.register_keras_serializable(package='Custom', name='VariationalAutoencoder')
class VariationalAutoencoder(Autoencoder):
    def __init__(self, input_shape, latent_dim, encoder_layer_sizes=[], decoder_layer_sizes=[], name='variational_autoencoder'):
        super(VariationalAutoencoder, self).__init__(input_shape, latent_dim, encoder_layer_sizes, decoder_layer_sizes, name)
        
        self.encoder = tf.keras.Sequential(
            [tf.keras.layers.Input(input_shape)] +
            [tf.keras.layers.Flatten()] +
            [tf.keras.layers.Dense(sz, activation='relu') for sz in encoder_layer_sizes] +
            [tf.keras.layers.Dense(latent_dim * 2)], name=f'{name}_encoder'  # output mean & log_var
        )
        
        self.decoder = tf.keras.Sequential(
            [tf.keras.layers.Input((latent_dim,))] +
            [tf.keras.layers.Dense(sz, activation='relu') for sz in decoder_layer_sizes] +
            [tf.keras.layers.Dense(np.prod(input_shape), activation='softplus'),
             tf.keras.layers.Reshape(input_shape)], name=f'{name}_decoder'
        )
        
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name='reconstruction_loss')
    
    def reparameterize(self, mean, log_var):
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon

    def encode(self, x):
        encoder_output = self.encoder(x)
        mean, log_var = tf.split(encoder_output, num_or_size_splits=2, axis=-1)
        return mean, log_var

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mean, log_var
    
    def compress(self, x):
        encoder_output = self.encoder(x)
        mean, log_var = tf.split(encoder_output, num_or_size_splits=2, axis=-1)
        
        # check this
        eps = tf.random.normal(shape=tf.shape(mean))
        sigma = tf.exp(0.5 * log_var)
        sample = mean + sigma * eps
        
        return sample
        
    
    def train_step(self, data):
        x = data[0]
        x = tf.cast(x, tf.float32)

        with tf.GradientTape() as tape:
            reconstructed, mean, log_var = self(x, training=True)
            loss = vae_loss(x, reconstructed, mean, log_var)
            r_loss = reconstruction_loss(x, reconstructed)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(r_loss)
        
        return {'loss': self.loss_tracker.result(), 'reconstruction_loss': self.reconstruction_loss_tracker.result()} 

    def test_step(self, data):
        x = data[0]
        x = tf.cast(x, tf.float32)

        reconstructed, mean, log_var = self(x, training=False)
        loss = vae_loss(x, reconstructed, mean, log_var)
        r_loss = reconstruction_loss(x, reconstructed)

        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(r_loss)
        
        return {'loss': self.loss_tracker.result(), 'reconstruction_loss': self.reconstruction_loss_tracker.result()}
    
    @property
    def metrics(self):
        return [self.loss_tracker, self.reconstruction_loss_tracker]

    @classmethod
    def from_config(self, config):
        return VariationalAutoencoder(**config)
    
    def get_config(self):
        return {
            'input_shape': self.input_shape,
            'latent_dim': self.latent_dim,
            'encoder_layer_sizes': self.encoder_layer_sizes,
            'decoder_layer_sizes': self.decoder_layer_sizes
        }
        
    def get_data_size_reduction(self):
        return np.prod(self.input_shape) / (self.latent_dim * 2)

@tf.keras.utils.register_keras_serializable(package='Custom', name='VAELoss')
def vae_loss(x, reconstructed, mean, log_var):
    r_loss = reconstruction_loss(x, reconstructed)
    kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
    return r_loss + kl_loss * 0.05

class VQVariationalAutoencoder(Encoder):
    def __init__(
        self, 
        input_shape,
        latent_dim,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        encoder_layer_sizes=[],
        decoder_layer_sizes=[],
        name='vq_variational_autoencoder'
    ):
        super().__init__(name=name)
        
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.encoder_layer_sizes = encoder_layer_sizes
        self.decoder_layer_sizes = decoder_layer_sizes

        self.encoder = tf.keras.Sequential(
            [tf.keras.layers.Input(input_shape),
             tf.keras.layers.Reshape((input_shape[0], input_shape[1]))] +
            [tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(sz, activation='relu')
            ) for sz in encoder_layer_sizes] +
            [tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(latent_dim * 2)
            )],
            name=f'{name}_encoder'
        )

        self.flatten_decoder_input = True
        self.decoder = tf.keras.Sequential(
            [tf.keras.layers.Input((input_shape[0] * latent_dim,))] +
            [tf.keras.layers.Dense(sz, activation='relu') for sz in decoder_layer_sizes] +
            [tf.keras.layers.Dense(np.prod(input_shape), activation='softplus'),
             tf.keras.layers.Reshape(input_shape)],
            name=f'{name}_decoder'
        )

        self.codebook = self.add_weight(
            name='codebook',
            shape=(num_embeddings, embedding_dim),
            initializer='uniform',
            trainable=True
        )
        
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")

    def build(self, input_shape):
        self.input_shape = input_shape
        super().build(input_shape)
        
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()

    def encode(self, x):
        x_encoded = self.encoder(x)
        mean, log_var = tf.split(x_encoded, num_or_size_splits=2, axis=-1)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * eps

    def decode(self, z):
        return self.decoder(z)

    def call(self, x, training=False):
        mean, log_var = self.encode(x)
        z_e = self.reparameterize(mean, log_var)
        z_e_expand = tf.expand_dims(z_e, 2)
        codebook_expand = tf.reshape(
            self.codebook, [1, 1, self.num_embeddings, self.embedding_dim]
        )
        distances = tf.reduce_sum((z_e_expand - codebook_expand) ** 2, axis=-1)
        encoding_indices = tf.argmin(distances, axis=-1)
        z_q = tf.gather(self.codebook, encoding_indices, axis=0)
        z_q_st = z_e + tf.stop_gradient(z_q - z_e)
        if self.flatten_decoder_input:
            z_q_st_flat = tf.reshape(z_q_st, [tf.shape(z_q_st)[0], -1])
            reconstructed = self.decode(z_q_st_flat)
        else:
            reconstructed = self.decode(z_q_st)
        vq_loss = tf.reduce_mean(tf.square(tf.stop_gradient(z_e) - z_q))
        return reconstructed, mean, log_var, vq_loss

    def train_step(self, data):
        x = data[0]
        x = tf.cast(x, tf.float32)
        with tf.GradientTape() as tape:
            reconstructed, mean, log_var, vq_loss = self(x, training=True)
            loss = vqvae_loss(x, reconstructed, mean, log_var, vq_loss, self.commitment_cost)
            r_loss = reconstruction_loss(x, reconstructed)
        gradients = tape.gradient(loss, self.trainable_variables)
        # gradients = [
        #     tf.clip_by_value(grad, -1.0, 1.0) if grad is not None else None
        #     for grad in gradients
        # ]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(r_loss)
        return {
            'loss': self.loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result()
        }
    
    def test_step(self, data):
        x = data[0]
        x = tf.cast(x, tf.float32)
        reconstructed, mean, log_var, vq_loss = self(x, training=False)
        loss = vqvae_loss(x, reconstructed, mean, log_var, vq_loss, self.commitment_cost)
        r_loss = reconstruction_loss(x, reconstructed)
        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(r_loss)
        return {
            'loss': self.loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result()
        }
        
    def encode_to_indices_probabilistic(self, x):
        mean, log_var = self.encode(x)

        eps = tf.random.normal(shape=tf.shape(mean))
        sigma = tf.exp(0.5 * log_var)
        z_e = mean + sigma * eps

        z_e_expand = tf.expand_dims(z_e, axis=2)
        codebook_expand = tf.reshape(self.codebook, [1, 1, self.num_embeddings, self.embedding_dim])

        distances = tf.reduce_sum((z_e_expand - codebook_expand) ** 2, axis=-1)
        encoding_indices = tf.argmin(distances, axis=-1)

        return encoding_indices

    def encode_to_codebook_vectors(self, x):
        mean, log_var = self.encode(x)
        
        eps = tf.random.normal(shape=tf.shape(mean))
        sigma = tf.exp(0.5 * log_var)
        z_e = mean + sigma * eps
        
        z_e_expand = tf.expand_dims(z_e, axis=2)
        codebook_expand = tf.reshape(self.codebook, [1, 1, self.num_embeddings, self.embedding_dim])
        
        distances = tf.reduce_sum((z_e_expand - codebook_expand) ** 2, axis=-1)
        encoding_indices = tf.argmin(distances, axis=-1)
        
        z_q = tf.gather(self.codebook, encoding_indices, axis=0)
        
        return z_q
        
    def compress(self, x):
        return self.encode_to_indices_probabilistic(x)
    
    @property
    def metrics(self):
        return [self.loss_tracker, self.reconstruction_loss_tracker]

    def get_config(self):
        return {
            'input_shape': self.input_shape,
            'latent_dim': self.latent_dim,
            'num_embeddings': self.num_embeddings,
            'embedding_dim': self.embedding_dim,
            'commitment_cost': self.commitment_cost,
            'encoder_layer_sizes': self.encoder_layer_sizes,
            'decoder_layer_sizes': self.decoder_layer_sizes
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def get_data_size_reduction(self):
        input_size = np.prod(self.input_shape)
        latent_size = self.input_shape[0] * (np.log2(self.num_embeddings) / 32)
        return input_size / latent_size
    

class GraphVQVariationalAutoencoder(VQVariationalAutoencoder):
    def __init__(
        self,
        input_shape,
        latent_dim,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        adjacency_matrix,
        encoder_layer_sizes=[],
        decoder_layer_sizes=[],
        name='graph_vq_variational_autoencoder'
    ):
        super().__init__(
            input_shape,
            latent_dim,
            num_embeddings,
            embedding_dim,
            commitment_cost,
            encoder_layer_sizes,
            decoder_layer_sizes,
            name
        )
        
        print(adjacency_matrix)
        
        adjacency_matrix = normalized_laplacian(adjacency_matrix)
        adjacency_matrix = rescale_laplacian(adjacency_matrix, lmax=2)
        
        print(adjacency_matrix)
        
        self.adjacency_matrix = adjacency_matrix
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.encoder_layer_sizes = encoder_layer_sizes
        self.decoder_layer_sizes = decoder_layer_sizes
        
        dropout_rate = 0.2
        use_batchnorm = True
        
        self.encoder_layers = []
        for units in encoder_layer_sizes:
            self.encoder_layers.append(spektral.layers.GCNConv(units, activation='relu'))
            if dropout_rate > 0:
                self.encoder_layers.append(tf.keras.layers.Dropout(dropout_rate))
            if use_batchnorm:
                self.encoder_layers.append(tf.keras.layers.BatchNormalization())

        self.encoder_layers.append(spektral.layers.GCNConv(latent_dim * 2))
        
        self.decoder_layers = []
        for units in decoder_layer_sizes:
            self.decoder_layers.append(spektral.layers.GCNConv(units, activation='relu'))
            if dropout_rate > 0:
                self.decoder_layers.append(tf.keras.layers.Dropout(dropout_rate))
            if use_batchnorm:
                self.decoder_layers.append(tf.keras.layers.BatchNormalization())
        
        self.decoder_layers.append(spektral.layers.GCNConv(input_shape[1], activation='softplus'))
        
        # self.encoder_layers = [spektral.layers.GCNConv(sz, activation='relu') for sz in encoder_layer_sizes] + [spektral.layers.GCNConv(latent_dim * 2)]
        # self.decoder_layers = [spektral.layers.GCNConv(sz, activation='relu') for sz in decoder_layer_sizes] + [spektral.layers.GCNConv(input_shape[1], activation='softplus')]
        
        self.flatten_decoder_input = False

        self.codebook = self.add_weight(
            name='codebook',
            shape=(num_embeddings, embedding_dim),
            initializer='uniform',
            trainable=True
        )
        
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        
    def encode(self, x):
        for layer in self.encoder_layers:
            x = layer([x, self.adjacency_matrix])
        mean, log_var = tf.split(x, num_or_size_splits=2, axis=-1)
        return mean, log_var
    
    def decode(self, z):
        for layer in self.decoder_layers:
            z = layer([z, self.adjacency_matrix])
        return z
    
    def summary(self):
        print(f'\n\033[1mModel: {self.name}\033[0m')
        GraphVQVariationalAutoencoder.print_summary_table('Encoder', self.encoder_layers)
        GraphVQVariationalAutoencoder.print_summary_table('Decoder', self.decoder_layers)
        print()
    
    def print_summary_table(name, layers):
        print(f'\n{name}:')
        print(f'{"Index":<6} {"Layer Name":<20} {"Output Shape":<20} {"Num Parameters":<15}')
        print('-' * 63)
        for i, layer in enumerate(layers, start=len(layers)):
            print(f'{i:<6} {layer.name:<20} {"?":<20} {layer.count_params():<15}')
        


@tf.keras.utils.register_keras_serializable(package='Custom', name='VQVAELoss')
def vqvae_loss(x, reconstructed, mean, log_var, vq_loss, commitment_cost):
    r_loss = reconstruction_loss(x, reconstructed)
    kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
    
    return r_loss + commitment_cost * vq_loss + kl_loss * 0.01


def gaussian_blur_3d(x, kernel_size=3, sigma=1.0):
    kernel = np.fromfunction(
        lambda x, y, z: (1/(2*np.pi*sigma**2)) * np.exp(-((x - kernel_size//2)**2 + (y - kernel_size//2)**2 + (z - kernel_size//2)**2) / (2*sigma**2)),
        (kernel_size, kernel_size, kernel_size)
    )
    kernel = kernel / np.sum(kernel)
    kernel = kernel[:, :, :, np.newaxis, np.newaxis]
    
    x = tf.nn.conv3d(x, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    
    return x[0, :, :, :, 0]
    
    
def train_models(models, losses, optimizers, data_generator, validation_data_generator, epochs=10, steps_per_epoch=100, batch_size=128, use_checkpoints=False):
    for model, loss, optimizer in zip(models, losses, optimizers):
        model.compile(optimizer=optimizer, loss=loss)
        model.build(next(iter(data_generator))[0].shape[1:])
        
        batch_x, batch_y = next(iter(data_generator))
        model.train_on_batch(batch_x, batch_y)
        
        model.summary()

        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, directory=f'saved_{model.name}_ckpt', max_to_keep=3)
        
        if ckpt_manager.latest_checkpoint and use_checkpoints:
            ckpt.restore(ckpt_manager.latest_checkpoint).assert_existing_objects_matched()
            print('restored from checkpoint:', ckpt_manager.latest_checkpoint)
        elif use_checkpoints:
            print('no checkpoint found--initializing from scratch.')

        checkpoint_callback = CheckpointCallback(ckpt_manager)
        callbacks = [checkpoint_callback] if use_checkpoints else []
        
        history = model.fit(
            data_generator,
            epochs=epochs,
            verbose=1,
            validation_data=validation_data_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=4,
            callbacks=callbacks
        )
        

def load_models_from_checkpoint(models, losses, optimizers, data_generator):
    for model, loss, optimizer in zip(models, losses, optimizers):
        model.compile(optimizer=optimizer, loss=loss)
        model.build(next(iter(data_generator))[0].shape[1:])
        model.summary()
        
        batch_x, batch_y = next(iter(data_generator))
        model.train_on_batch(batch_x, batch_y)
        
        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, directory=f'saved_{model.name}_ckpt', max_to_keep=3)
        
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).assert_existing_objects_matched()
            print('restored from checkpoint:', ckpt_manager.latest_checkpoint)
        else:
            print('no checkpoint found--initializing from scratch.')

        
def main():
    
    X, XC, C, PXC, EXC = load_SS_dataset('../dSSdMS/dSS_20241117_gaussgass_700samplearea7000_1.0e+04events_random_centered.npz')
    
    train_split = int(0.9 * X.shape[0])
    
    N = 4
    
    batch_size = 64
    data_generator = N_channel_scatter_events_autoencoder_generator(XC[:train_split], max_N=N, batch_size=batch_size)
    validation_data_generator = N_channel_scatter_events_autoencoder_generator(XC[train_split:], max_N=N, batch_size=batch_size)
    
    input_shape =  (XC.shape[1] * XC.shape[2], XC.shape[3])
    latent_dim = 1280
    num_embeddings = 64
    embedding_dim = 128
    commitment_cost = 0.20
    adjacency_matrix = create_grid_adjacency(XC.shape[1])
    
    autoencoder = Autoencoder(input_shape, latent_dim, encoder_layer_sizes=[1280, 1280], decoder_layer_sizes=[1280, 1280])
    variational_autoencoder = VariationalAutoencoder(input_shape, latent_dim, encoder_layer_sizes=[256, 256], decoder_layer_sizes=[256, 256])
    vq_variational_autoencoder = VQVariationalAutoencoder(input_shape, latent_dim, 256, embedding_dim=embedding_dim, commitment_cost=commitment_cost, encoder_layer_sizes=[128, 128], decoder_layer_sizes=[256, 256])
    graph_vq_variational_autoencoder = GraphVQVariationalAutoencoder(input_shape, 700, num_embeddings, embedding_dim=700, commitment_cost=commitment_cost, adjacency_matrix=adjacency_matrix, encoder_layer_sizes=[700, 700], decoder_layer_sizes=[700, 700])
    
    models = [autoencoder]#, variational_autoencoder, vq_variational_autoencoder, graph_vq_variational_autoencoder]
    losses = [reconstruction_loss]#, vae_loss, vqvae_loss, vqvae_loss]
    optimizers = [tf.keras.optimizers.Adam(learning_rate=5e-4)]#, tf.keras.optimizers.Adam(learning_rate=5e-4), tf.keras.optimizers.Adam(learning_rate=5e-4), tf.keras.optimizers.Adam(learning_rate=5e-4)]
    
    train_models(models, losses, optimizers, data_generator, validation_data_generator, batch_size=batch_size, epochs=30, steps_per_epoch=64, use_checkpoints=False)
    
    test_events, _ = next(iter(validation_data_generator))
    
    test_event = test_events[0].reshape((XC.shape[1], XC.shape[2], XC.shape[3]))
    convolved_test_event = gaussian_blur_3d(test_event[np.newaxis, :, :, :, np.newaxis], kernel_size=5, sigma=1)
    autoencoder_reconstruction = autoencoder(test_events)[0].numpy().reshape((XC.shape[1], XC.shape[2], XC.shape[3]))
    variational_autoencoder_reconstruction = variational_autoencoder(test_events)[0][0].numpy().reshape((XC.shape[1], XC.shape[2], XC.shape[3]))
    vq_variational_autoencoder_reconstruction = vq_variational_autoencoder(test_events)[0][0].numpy().reshape((XC.shape[1], XC.shape[2], XC.shape[3]))
    graph_vq_variational_autoencoder_reconstruction = graph_vq_variational_autoencoder(test_events)[0][0].numpy().reshape((XC.shape[1], XC.shape[2], XC.shape[3]))
    
    display_events = [test_event, convolved_test_event, autoencoder_reconstruction, variational_autoencoder_reconstruction, vq_variational_autoencoder_reconstruction, graph_vq_variational_autoencoder_reconstruction]
    display_events_titles = [
        'Original', 
        'Original (Gaussian blur, σ=1)', 
        f'Autoencoder\nData size reduction: {int(autoencoder.get_data_size_reducton())}x\nReconstruction loss (MSE): {reconstruction_loss(test_event, autoencoder_reconstruction).numpy() : .3f} phd^2', 
        f'Variational Autoencoder\nData size reduction: {int(variational_autoencoder.get_data_size_reducton())}x\nReconstruction loss (MSE): {reconstruction_loss(test_event, variational_autoencoder_reconstruction).numpy() : .3f} phd^2', 
        f'Vector-Quantized Variational Autoencoder\nData size reduction: {int(vq_variational_autoencoder.get_data_size_reduction())}x\nReconstruction loss (MSE): {reconstruction_loss(test_event, vq_variational_autoencoder_reconstruction).numpy() : .3f} phd^2',
        f'Graph Vector-Quantized Variational Autoencoder\nData size reduction: {int(graph_vq_variational_autoencoder.get_data_size_reduction())}x\nReconstruction loss (MSE): {reconstruction_loss(test_event, graph_vq_variational_autoencoder_reconstruction).numpy() : .3f} phd^2'
    ]
    
    plot_events(np.stack(display_events, axis=0), title='', subtitles=display_events_titles)
    
    '''
    aux_data_generator = N_channel_scatter_events_generator(XC[:train_split], C[:train_split], PXC[:train_split], max_N=N, batch_size=batch_size, y='N')
    aux_val_data_generator = N_channel_scatter_events_generator(XC[train_split:], C[train_split:], PXC[train_split:], max_N=N, batch_size=batch_size, y='N')
    adjacency_matrix = create_grid_adjacency(XC.shape[1])
    
    def aux_model_build_compile(modelType, args, input_shape, loss=tf.keras.losses.MeanSquaredError(), metrics=[]):
        model = modelType(*args)
        model.build(input_shape)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4), 
            loss=loss,
            metrics=metrics
        )
        return model
    
    raw_aux_model_1 = aux_model_build_compile(GATNumScattersModel, [adjacency_matrix], (None, XC.shape[1], XC.shape[2]))
    compressed_aux_model_1 = aux_model_build_compile(DenseNumScattersModel, [], (None, latent_dim))
    compressed_aux_model_2 = aux_model_build_compile(DenseNumScattersModel, [], (None, latent_dim))
    compressed_aux_model_3 = aux_model_build_compile(GATNumScattersModel, [adjacency_matrix], (None, XC.shape[1], latent_dim))
    
    run_aux_task(
        [raw_aux_model_1, compressed_aux_model_1, compressed_aux_model_3],
        [None, autoencoder.compress, vq_variational_autoencoder.encode_to_codebook_vectors],
        labels = ['Raw data', 'AE latent', 'VQ-VAE codebook vectors'],
        data_generator=aux_data_generator, 
        val_data_generator=aux_val_data_generator,
        fname_suffix='_models'
    )
    
    aux_data_generator = N_channel_scatter_events_generator(XC[:train_split], C[:train_split], PXC[:train_split], max_N=N, batch_size=batch_size, y='XYZ')
    aux_val_data_generator = N_channel_scatter_events_generator(XC[train_split:], C[train_split:], PXC[train_split:], max_N=N, batch_size=batch_size, y='XYZ')
    
    loss = GATMultivariateNormalModel.combined_loss
    metrics = [
        GATMultivariateNormalModel.pdf_loss,
        GATMultivariateNormalModel.mask_loss,
    ]
    
    raw_aux_model_1 = aux_model_build_compile(GATMultivariateNormalModel, [adjacency_matrix], (None, XC.shape[1], XC.shape[2]), loss=loss, metrics=metrics)
    compressed_aux_model_1 = aux_model_build_compile(DenseMultivariateNormalModel, [], (None, latent_dim), loss=loss, metrics=metrics)
    compressed_aux_model_2 = aux_model_build_compile(DenseMultivariateNormalModel, [], (None, latent_dim), loss=loss, metrics=metrics)
    compressed_aux_model_3 = aux_model_build_compile(GATMultivariateNormalModel, [adjacency_matrix], (None, XC.shape[1], latent_dim), loss=loss, metrics=metrics)
    
    run_aux_task(
        [raw_aux_model_1, compressed_aux_model_1, compressed_aux_model_2, compressed_aux_model_3],
        [None, autoencoder.compress, vq_variational_autoencoder.compress, vq_variational_autoencoder.encode_to_codebook_vectors],
        labels = ['Raw data', 'AE latent', 'VQ-VAE codebook indices', 'VQ-VAE codebook vectors'],
        data_generator=aux_data_generator, 
        val_data_generator=aux_val_data_generator,
        fname_suffix='_models'
    )
    
    raw_aux_model_1 = aux_model_build_compile(GATMultivariateNormalModel, [adjacency_matrix], (None, XC.shape[1], XC.shape[2]), loss=loss, metrics=metrics)
    raw_aux_model_2 = aux_model_build_compile(GATMultivariateNormalModel, [adjacency_matrix], (None, XC.shape[1], XC.shape[2]), loss=loss, metrics=metrics)
    compressed_aux_model_1 = aux_model_build_compile(DenseMultivariateNormalModel, [], (None, latent_dim), loss=loss, metrics=metrics)
    
    run_aux_task(
        [raw_aux_model_1, compressed_aux_model_1, raw_aux_model_2],
        [None, autoencoder.encode, lambda x: autoencoder.decode(autoencoder.encode(x))],
        labels = ['Raw data', 'AE latent', 'AE reconstruction'],
        data_generator=aux_data_generator,
        val_data_generator=aux_val_data_generator,
        fname_suffix='_single_model'
    )
    
    '''
    X_test, XC_test, Y_test, C_test, P_test, E_test = generate_N_channel_scatter_events(X, XC, C, PXC, EXC, max_N=4, num_events=32768, normalize=True)
    
    # codebook_usage_histogram(vq_variational_autoencoder, XC_test[:256])
    
    fit_model = TSNE(n_components=2, perplexity=250, random_state=42)

    latent = np.concatenate([autoencoder.compress(XC_test[i:i + 256]).numpy() for i in tqdm(range(0, len(XC_test), 256))], axis=0)
    latent_space = fit_model.fit_transform(latent)
    
    vis_latent_space_footprint(fit_model, latent_space, XC_test)
    vis_latent_space_num_scatters(fit_model, latent_space, Y_test, N=4)
    vis_latent_space_phd(fit_model, latent_space, XC_test)
    
    
    
    
    
        
    

if __name__ == '__main__':
    set_mpl_style()
    main()