import tensorflow as tf
import tensorflow.keras.backend as K
import spektral
from spektral.utils import normalized_laplacian, rescale_laplacian

import re
import os
import seaborn as sns
from scipy import stats
import scipy.sparse as sp
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
import multiprocessing as mp
from functools import partial

from preprocessing import *
from pulse import *
from generator import *

from regression_models import GATMultivariateNormalModel, DenseMultivariateNormalModel
from regression_models import GATNumScattersModel, DenseNumScattersModel

from simple_pulse import vertex_electron_batch_generator, Params
from simple_likelihood import compute_mle_vertex_positions


def localpooling_filter(adj,
                        symmetric: bool = True,
                        add_self_loops: bool = True,
                        dtype=np.float32):
    """
    Build the normalised adjacency used by Kipf & Welling’s GCN
        Ā = D^{-1/2} (A + I) D^{-1/2}   (symmetric)
        Ā = D^{-1}   (A + I)            (random‑walk)

    Parameters
    ----------
    adj : scipy.sparse.spmatrix | np.ndarray
        Unnormalised adjacency (shape [N, N]).
    symmetric : bool, default True
        Use symmetric (GCN) normalisation.  
        Set False for the left‑normalised random‑walk variant.
    add_self_loops : bool, default True
        Whether to add self‑loops before normalising.
    dtype : np.dtype, default np.float32
        dtype of the returned matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        Normalised adjacency in CSR format, dtype ``dtype``.
    """
    # 1. Make sure we have a CSR sparse matrix
    if sp.isspmatrix(adj):
        A = adj.tocsr().astype(dtype)
    else:                                   # dense → sparse
        A = sp.csr_matrix(adj, dtype=dtype)

    # 2. Optionally add self‑loops
    if add_self_loops:
        A = A + sp.eye(A.shape[0], dtype=dtype, format="csr")

    # 3. Degree vector (with ε‑guard against isolated nodes)
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0.0] = 1.0                  # avoids division by zero

    # 4. Normalise
    if symmetric:
        deg_inv_sqrt = 1.0 / np.sqrt(deg)
        D_inv_sqrt   = sp.diags(deg_inv_sqrt)
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    else:                                  # random‑walk
        deg_inv = 1.0 / deg
        D_inv   = sp.diags(deg_inv)
        A_norm  = D_inv @ A

    # 5. Return CSR float32/float64, as requested
    return A_norm.tocsr().astype(dtype)


@tf.keras.utils.register_keras_serializable(package='Custom', name='AutoencoderLoss')
def reconstruction_loss(x, reconstructed, **kwargs):
    return tf.reduce_mean(tf.keras.losses.MeanSquaredError()(x, reconstructed))

@tf.keras.utils.register_keras_serializable(package='Custom', name='EMDAutoencoderLoss')
def emd_reconstruction_loss(x, reconstructed, **kwargs):
    x = tf.reshape(x, (tf.shape(x)[0], 24, 24, 700))
    reconstructed = tf.reshape(reconstructed, (tf.shape(reconstructed)[0], 24, 24, 700))

    x_cumsum_x = tf.cumsum(x, axis=1)
    x_cumsum_xy = tf.cumsum(x_cumsum_x, axis=2)
    x_cumsum_xyz = tf.cumsum(x_cumsum_xy, axis=3)

    reconstructed_cumsum_x = tf.cumsum(reconstructed, axis=1)
    reconstructed_cumsum_xy = tf.cumsum(reconstructed_cumsum_x, axis=2)
    reconstructed_cumsum_xyz = tf.cumsum(reconstructed_cumsum_xy, axis=3)

    emd_xyz = tf.reduce_mean(tf.abs(x_cumsum_xyz - reconstructed_cumsum_xyz))
    
    return emd_xyz

@tf.keras.utils.register_keras_serializable(package='Custom', name='CumulativeAutoencoderLoss')
def combined_reconstruction_loss(x, reconstructed, emd_weight=0.0, sum_weight=1e-7, **kwargs):
    e_loss = emd_reconstruction_loss(x, reconstructed)
    r_loss = reconstruction_loss(x, reconstructed)
    s_loss = tf.abs(tf.reduce_sum(x) - tf.reduce_sum(reconstructed))
    return r_loss + emd_weight * e_loss + sum_weight * s_loss

@tf.keras.utils.register_keras_serializable(package='Custom', name='VAELoss')
def vae_loss(x, reconstructed, mean, log_var):
    r_loss = reconstruction_loss(x, reconstructed)
    kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
    return r_loss + kl_loss * 0.05

@tf.keras.utils.register_keras_serializable(package='Custom', name='VQVAELoss')
def vqvae_loss(x, reconstructed, mean, log_var, vq_loss, commitment_cost):
    r_loss = reconstruction_loss(x, reconstructed) * 10
    kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
    
    return r_loss + commitment_cost * vq_loss + kl_loss * 0.01


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

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
    
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

    @property
    def metrics(self):
        return [self.loss_tracker, self.reconstruction_loss_tracker]

    def train_step(self, data):
        x = data[0]
        x = tf.cast(x, tf.float32)

        with tf.GradientTape() as tape:
            reconstructed = self(x, training=True)
            loss = self.loss(x, reconstructed)
            r_loss = reconstruction_loss(x, reconstructed)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.reconstruction_loss_tracker.update_state(r_loss)
        self.loss_tracker.update_state(loss)
        
        return {
            'loss': self.loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
        }
    
    def test_step(self, data):
        x = data[0]
        x = tf.cast(x, tf.float32)
        reconstructed = self(x, training=False)
        r_loss = reconstruction_loss(x, reconstructed)
        self.reconstruction_loss_tracker.update_state(r_loss)
        self.loss_tracker.update_state(r_loss)

        return {
            'loss': self.loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
        }

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
        self.emd_reconstruction_loss_tracker = tf.keras.metrics.Mean(name="emd_reconstruction_loss")
    
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
            emd_loss = emd_reconstruction_loss(x, reconstructed)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(r_loss)
        self.emd_reconstruction_loss_tracker.update_state(emd_loss)
        
        return {
            'loss': self.loss_tracker.result(), 
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'emd_reconstruction_loss': self.emd_reconstruction_loss_tracker.result()
        }

    def test_step(self, data):
        x = data[0]
        x = tf.cast(x, tf.float32)
        reconstructed, mean, log_var = self(x, training=False)
        r_loss = reconstruction_loss(x, reconstructed)
        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
        
        # Calculate total loss
        loss = r_loss + kl_loss * 0.05
        
        # Update metrics
        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(r_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            'loss': self.loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result()
        }
    
    @property
    def metrics(self):
        return [self.loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

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


class GraphVariationalAutoencoder(tf.keras.Model):
    def __init__(
        self, 
        input_shape, 
        latent_dim, 
        encoder_layer_sizes=[], 
        decoder_layer_sizes=[], 
        pooling=True,
        adjacency_matrix=None,
        name='graph_variational_autoencoder2'
    ):
        super().__init__(name=name)
        
        self.latent_dim = latent_dim
        self.pooling = pooling

        # Preprocess adjacency
        adjacency_matrix = normalized_laplacian(adjacency_matrix)
        adjacency_matrix = rescale_laplacian(adjacency_matrix, lmax=2)
        self.adjacency_matrix = adjacency_matrix
        
        # Encoder layers
        self.encoder_layers = []
        self.pooling_layers = []
        
        # Add GCN + pooling
        nodes = input_shape[0]
        for units in encoder_layer_sizes:
            self.encoder_layers.append(spektral.layers.GCNConv(units, activation='relu'))
            if pooling:
                nodes //= 2
                self.pooling_layers.append(spektral.layers.TopKPool(ratio=0.5, return_selection=True))
        
        # Final encoder layer -> produce mean+log_var
        self.encoder_layers.append(spektral.layers.GCNConv(latent_dim * 2))
        
        # Decoder layers
        self.decoder_layers = []
        self.unpooling_layers = []
        
        # Initial decoder layer
        self.decoder_layers.append(spektral.layers.GCNConv(decoder_layer_sizes[0], activation='relu'))
        
        # Add GCN + unpooling
        for units in decoder_layer_sizes[1:]:
            if pooling:
                self.unpooling_layers.append(TopKUnpool())
            self.decoder_layers.append(spektral.layers.GCNConv(units, activation='relu'))
        
        # Final decoder layer
        self.decoder_layers.append(spektral.layers.GCNConv(input_shape[1], activation='softplus'))
        
        # Metric trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.pooling_loss_tracker = tf.keras.metrics.Mean(name="pooling_loss")

    def reparameterize(self, mean, log_var):
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * eps

    # -------------------------------------------------------------------------
    # Encode a single sample (shape: [1, n_nodes, n_features])
    # -------------------------------------------------------------------------
    def encode_single(self, x_single, training=False):
        """
        Encode one sample at a time.
        Returns mean, log_var, plus adjacency & selection info for decode.
        """
        # We'll keep track of adjacency and selection per-layers
        adjacency_stack = []
        selection_stack = []
        
        # Start with adjacency for this sample (identical if all samples share adjacency)
        # We can simply use self.adjacency_matrix for each sample:
        current_adj = self.adjacency_matrix  # shape [n_nodes, n_nodes]
        
        x_enc = x_single[0]
        adjacency_stack.append(current_adj)  # store the adjacency used

        # 2) Apply subsequent GCN + (optional) pooling
        idx_pool_layer = 0
        for layer in self.encoder_layers[:-1]:  # up to the final layer
            x_enc = layer([x_enc, current_adj], training=training)

            if self.pooling and idx_pool_layer < len(self.pooling_layers):
                pool_layer = self.pooling_layers[idx_pool_layer]
                idx_pool_layer += 1
                
                # Because Spektral's pools expect adjacency as a sparse
                adj_sparse = tf.sparse.from_dense(current_adj)
                pool_layer._n_nodes = tf.shape(current_adj)[0]
                x_enc, adj_pooled, selection_idx = pool_layer([x_enc, adj_sparse], training=training)
                current_adj = tf.sparse.to_dense(adj_pooled)

                adjacency_stack.append(current_adj)
                selection_stack.append(selection_idx)
            else:
                print('No pooling layer:', layer.name)

        # 3) Final layer to produce mean+log_var
        final_layer = self.encoder_layers[-1]
        x_enc = final_layer([x_enc, current_adj], training=training)

        mean, log_var = tf.split(x_enc, num_or_size_splits=2, axis=-1)  # shape: [1, n_nodes, latent_dim] if not pooled
        # For a typical "global" embedding, you might reduce_mean across nodes or something. 
        # But let's assume you want the node-level embedding as is.

        return mean, log_var, adjacency_stack, selection_stack

    # -------------------------------------------------------------------------
    # Decode a single sample from z, plus adjacency/selection info
    # -------------------------------------------------------------------------
    def decode_single(self, z_single, adjacency_stack, selection_stack, training=False):
        """
        Decode one sample at a time, reversing the encode steps.
        z_single: shape [1, n_nodes, latent_dim]
        adjacency_stack, selection_stack: from encode_single
        """
        # Start with the adjacency from the last encode step
        current_adj = adjacency_stack[-1]  # shape [?, ?]

        # 1) initial decoder layer
        x_dec = z_single

        # We'll iterate over the GCN/unpool pairs in reverse
        num_unpools = len(self.unpooling_layers)

        for i, gcn_layer in enumerate(self.decoder_layers[:-1]):
            x_dec = gcn_layer([x_dec, current_adj], training=training)

            if self.pooling and i < num_unpools:
                # matching unpool index: from the last selection
                unpool_layer = self.unpooling_layers[num_unpools - 1 - i]

                # adjacency index for unpool 
                # = adjacency_stack index used in forward pass
                adj_idx = len(adjacency_stack) - 2 - i
                sel_idx = len(selection_stack) - 1 - i

                if adj_idx >= 0 and sel_idx >= 0:
                    # fetch adjacency & selection
                    old_adj = adjacency_stack[adj_idx]
                    selection = selection_stack[sel_idx]

                    # reshape for layer call
                    x_dec = unpool_layer([tf.expand_dims(x_dec, 0), tf.expand_dims(selection, 0), tf.expand_dims(old_adj, 0)])
                    # unpool_layer returns shape [1, old_n_nodes, channels]
                    # so no further adjacency change from unpool
                    x_dec = tf.squeeze(x_dec, axis=0)  # shape [old_n_nodes, channels]
                    old_adj = tf.cast(old_adj, x_dec.dtype)
                    current_adj = old_adj  # revert adjacency
                else:
                    print("Unpool indices out of range - skipping unpool")


        # final decode GCN
        # use adjacency_stack[0] as original adjacency
        original_adj = adjacency_stack[0]
        x_dec = self.decoder_layers[-1]([x_dec, original_adj], training=training)
        return x_dec

    # -------------------------------------------------------------------------
    # call over the entire batch: we loop in Python
    # -------------------------------------------------------------------------
    def call(self, x, training=False):
        """
        x: shape [batch_size, n_nodes, n_features]
        We'll:
          1) loop over batch dimension in Python (range(batch_size))
          2) call encode_single(...) for each sample
          3) reparameterize for each sample
          4) call decode_single(...) for each sample
          5) collect outputs in a list/tensorarray
        """
        batch_size = tf.shape(x)[0]
        
        # We'll store the results in Python lists, then stack
        recons_list = []
        mean_list = []
        logvar_list = []

        for b in tf.range(batch_size):  
            # We do .numpy() so Python range(...) sees an integer 
            # (Caution: this means Eager must be enabled, which Keras typically does by default.)
            # If you want fully graph-based, you'd use tf.while_loop or tf.map_fn instead.

            x_single = x[b:b+1]  # shape [1, n_nodes, n_features]

            mean, log_var, adj_stack, sel_stack = self.encode_single(x_single, training=training)
            z_single = self.reparameterize(mean, log_var)
            reconstructed_single = self.decode_single(z_single, adj_stack, sel_stack, training=training)

            recons_list.append(reconstructed_single)
            mean_list.append(mean)
            logvar_list.append(log_var)

        # Stack them back
        reconstructed = tf.stack(recons_list, axis=0)  
        mean_out = tf.stack(mean_list, axis=0)
        logvar_out = tf.stack(logvar_list, axis=0)

        return reconstructed, mean_out, logvar_out

    # -------------------------------------------------------------------------
    # Standard train_step as before
    # -------------------------------------------------------------------------
    def train_step(self, data):
        x = data[0]
        x = tf.cast(x, tf.float32)
        
        with tf.GradientTape() as tape:
            reconstructed, mean, log_var = self(x, training=True)
            
            # Compute losses
            r_loss = reconstruction_loss(x, reconstructed)  
            kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))

            
            loss = r_loss + kl_loss * 0.05
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(r_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            'loss': self.loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        x = data[0]
        x = tf.cast(x, tf.float32)
        reconstructed, mean, log_var = self(x, training=False)
        r_loss = reconstruction_loss(x, reconstructed)
        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
        
        # Calculate total loss
        loss = r_loss + kl_loss * 0.05
        
        # Update metrics
        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(r_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            'loss': self.loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result()
        }

    @property
    def metrics(self):
        metrics = [self.loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]
        # if self.pooling:
        #     metrics.append(self.pooling_loss_tracker)
        return metrics

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
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.vq_loss_tracker = tf.keras.metrics.Mean(name="vq_loss")
        # self.emd_reconstruction_loss_tracker = tf.keras.metrics.Mean(name="emd_reconstruction_loss")

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
            emd_loss = emd_reconstruction_loss(x, reconstructed)
            kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
        gradients = tape.gradient(loss, self.trainable_variables)
        # gradients = [
        #     tf.clip_by_value(grad, -1.0, 1.0) if grad is not None else None
        #     for grad in gradients
        # ]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(r_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.vq_loss_tracker.update_state(vq_loss)
        # self.emd_reconstruction_loss_tracker.update_state(emd_loss)
        return {
            'loss': self.loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
            'vq_loss': self.vq_loss_tracker.result()
            # 'emd_reconstruction_loss': self.emd_reconstruction_loss_tracker.result()
        }
    
    def test_step(self, data):
        x = data[0]
        x = tf.cast(x, tf.float32)
        reconstructed, mean, log_var, vq_loss = self(x, training=False)
        loss = vqvae_loss(x, reconstructed, mean, log_var, vq_loss, self.commitment_cost)
        r_loss = reconstruction_loss(x, reconstructed)
        emd_loss = emd_reconstruction_loss(x, reconstructed)
        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(r_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.vq_loss_tracker.update_state(vq_loss)
        # self.emd_reconstruction_loss_tracker.update_state(emd_loss)
        return {
            'loss': self.loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
            'vq_loss': self.vq_loss_tracker.result()
            # 'emd_reconstruction_loss': self.emd_reconstruction_loss_tracker.result()
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
        return [self.loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker, self.vq_loss_tracker]

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
        name='deep_gcn_vq_variational_autoencoder'
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
        
        adjacency_matrix = normalized_laplacian(adjacency_matrix)
        adjacency_matrix = rescale_laplacian(adjacency_matrix, lmax=2)
        
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

        # Model based on DeepGCN
        # See original paper: https://arxiv.org/abs/1910.06849
        
        self.encoder_layers = []
        self.encoder_blocks = []
        
        self.dilated_adj_matrices = {}
        dilation_rates = [1, 2, 4, 8]
        for rate in dilation_rates:
            if rate == 1:
                self.dilated_adj_matrices[rate] = self.adjacency_matrix
            else:
                dilated_adj = self.adjacency_matrix
                for _ in range(rate - 1):
                    dilated_adj = tf.matmul(dilated_adj, self.adjacency_matrix)
                dilated_adj = normalized_laplacian(dilated_adj.numpy())
                dilated_adj = rescale_laplacian(dilated_adj, lmax=2)
                self.dilated_adj_matrices[rate] = dilated_adj
        
        if encoder_layer_sizes:
            self.encoder_layers.append(spektral.layers.GCNConv(encoder_layer_sizes[0], activation='relu'))
            
            for i, units in enumerate(encoder_layer_sizes[1:], 1):
                block_layers = []
                
                dilation_rate = dilation_rates[i % len(dilation_rates)]
                block_layers.append((spektral.layers.GCNConv(units, activation='relu'), dilation_rate))
                
                if dropout_rate > 0:
                    block_layers.append(tf.keras.layers.Dropout(dropout_rate))
                if use_batchnorm:
                    block_layers.append(tf.keras.layers.BatchNormalization())
                    
                self.encoder_blocks.append(block_layers)
        
        self.encoder_layers.append(spektral.layers.GCNConv(latent_dim * 2))
        
        self.decoder_layers = []
        self.decoder_blocks = []
        
        if decoder_layer_sizes:
            self.decoder_layers.append(spektral.layers.GCNConv(decoder_layer_sizes[0], activation='relu'))
            
            for i, units in enumerate(decoder_layer_sizes[1:], 1):
                block_layers = []
                
                dilation_rate = dilation_rates[i % len(dilation_rates)]
                block_layers.append((spektral.layers.GCNConv(units, activation='relu'), dilation_rate))
                
                if dropout_rate > 0:
                    block_layers.append(tf.keras.layers.Dropout(dropout_rate))
                if use_batchnorm:
                    block_layers.append(tf.keras.layers.BatchNormalization())
                    
                current_dim = units
                self.decoder_blocks.append(block_layers)

        self.decoder_layers.append(spektral.layers.GCNConv(input_shape[1], activation='softplus'))

        self.gan_layers = []
        self.gan_blocks = []

        if decoder_layer_sizes:
            self.gan_layers.append(spektral.layers.GCNConv(decoder_layer_sizes[0], activation='relu'))

            for i, units in enumerate(decoder_layer_sizes[1:], 1):
                block_layers = []
                dilation_rate = dilation_rates[i % len(dilation_rates)]
                block_layers.append((spektral.layers.GCNConv(units, activation='relu'), dilation_rate))

                if dropout_rate > 0:
                    block_layers.append(tf.keras.layers.Dropout(dropout_rate))
                if use_batchnorm:
                    block_layers.append(tf.keras.layers.BatchNormalization())
                
                self.gan_blocks.append(block_layers)

        self.flatten_decoder_input = False

        self.codebook = self.add_weight(
            name='codebook',
            shape=(num_embeddings, embedding_dim),
            initializer='uniform',
            trainable=True
        )
        
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.emd_reconstruction_loss_tracker = tf.keras.metrics.Mean(name="emd_reconstruction_loss")
        
    def encode(self, x):
        for layer in self.encoder_layers[:-1]:
            x = layer([x, self.adjacency_matrix])
            
        for block in self.encoder_blocks:
            residual = x
            for layer in block:
                if isinstance(layer, tuple) and isinstance(layer[0], spektral.layers.GCNConv):
                    gcn_layer, dilation_rate = layer
                    x = gcn_layer([x, self.dilated_adj_matrices[dilation_rate]])
                elif isinstance(layer, spektral.layers.GCNConv):
                    x = layer([x, self.adjacency_matrix])
                else:
                    x = layer(x)
            x = x + residual
        
        x = self.encoder_layers[-1]([x, self.adjacency_matrix])
        mean, log_var = tf.split(x, num_or_size_splits=2, axis=-1)
        return mean, log_var
    
    def decode(self, z):
        if len(self.decoder_layers) > 1:
            z = self.decoder_layers[0]([z, self.adjacency_matrix])
            
        for block in self.decoder_blocks:
            residual = z
            for layer in block:
                if isinstance(layer, tuple) and isinstance(layer[0], spektral.layers.GCNConv):
                    gcn_layer, dilation_rate = layer
                    z = gcn_layer([z, self.dilated_adj_matrices[dilation_rate]])
                elif isinstance(layer, spektral.layers.GCNConv):
                    z = layer([z, self.adjacency_matrix])
                else:
                    z = layer(z)
            z = z + residual
        
        z = self.decoder_layers[-1]([z, self.adjacency_matrix])
        return z
    
    def generate(self, z):
        if len(self.gan_layers) > 1:
            z = self.gan_layers[0]([z, self.adjacency_matrix])
            
        for block in self.gan_blocks:
            residual = z
            for layer in block:
                if isinstance(layer, tuple) and isinstance(layer[0], spektral.layers.GCNConv):
                    gcn_layer, dilation_rate = layer
                    z = gcn_layer([z, self.dilated_adj_matrices[dilation_rate]])
                elif isinstance(layer, spektral.layers.GCNConv):
                    z = layer([z, self.adjacency_matrix])
                else:
                    z = layer(z)
            z = z + residual
        
        if len(self.gan_layers) > 1:
            z = self.gan_layers[-1]([z, self.adjacency_matrix])
        return z
    
    def freeze_encoder_weights(self):
        for layer in self.encoder_layers:
            layer.trainable = False
    
    def unfreeze_encoder_weights(self):
        for layer in self.encoder_layers:
            layer.trainable = True
    
    def summary(self):
        print(f'\n\033[1mModel: {self.name}\033[0m')
        print("\nEncoder:")
        print(f'{"Layer":<30} {"Output Shape":<20} {"Params":<10}')
        print('-' * 60)
        
        for i, layer in enumerate(self.encoder_layers):
            print(f'{layer.name:<30} {"?":<20} {layer.count_params():<10}')
            if i < len(self.pooling_layers):
                print(f'{self.pooling_layers[i].name:<30} {"?":<20} {self.pooling_layers[i].count_params():<10}')
        
        print("\nDecoder:")
        print(f'{"Layer":<30} {"Output Shape":<20} {"Params":<10}')
        print('-' * 60)
        
        for i, layer in enumerate(self.decoder_layers):
            if i < len(self.unpooling_layers):
                print(f'{self.unpooling_layers[i].name:<30} {"?":<20} {self.unpooling_layers[i].count_params():<10}')
            print(f'{layer.name:<30} {"?":<20} {layer.count_params():<10}')


class SimpleGraphVQVariationalAutoencoder(VQVariationalAutoencoder):
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
        name='simple_vqvae'
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

        adjacency_matrix = normalized_laplacian(adjacency_matrix)
        adjacency_matrix = rescale_laplacian(adjacency_matrix, lmax=2)
        
        self.adjacency_matrix = adjacency_matrix
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.encoder_layer_sizes = encoder_layer_sizes
        self.decoder_layer_sizes = decoder_layer_sizes
        
        self.encoder_layers = [spektral.layers.GCNConv(sz, activation='relu') for sz in encoder_layer_sizes] + [spektral.layers.GCNConv(latent_dim * 2)]
        self.decoder_layers = [spektral.layers.GCNConv(sz, activation='relu') for sz in decoder_layer_sizes] + [spektral.layers.GCNConv(input_shape[1], activation='softplus')]
        
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
        SimpleGraphVQVariationalAutoencoder.print_summary_table('Encoder', self.encoder_layers)
        SimpleGraphVQVariationalAutoencoder.print_summary_table('Decoder', self.decoder_layers)
        print()
    
    def print_summary_table(name, layers):
        print(f'\n{name}:')
        print(f'{"Index":<6} {"Layer Name":<20} {"Output Shape":<20} {"Num Parameters":<15}')
        print('-' * 63)
        for i, layer in enumerate(layers, start=len(layers)):
            print(f'{i:<6} {layer.name:<20} {"?":<20} {layer.count_params():<15}')


class TopKUnpool(tf.keras.layers.Layer):
    """
    TopKUnpool layer for graph neural networks.
    This layer reverses the operation performed by TopKPool.
    
    Args:
        **kwargs: Additional arguments for the Layer class.
    """
    def __init__(self, **kwargs):
        super(TopKUnpool, self).__init__(**kwargs)
        self.supports_masking = False

    def build(self, input_shape):
        self.built = True

    @tf.function
    def call(self, inputs):
        """
        Implements the unpooling operation.
        
        Args:
            inputs: List containing:
                - x: Pooled node features tensor with shape [batch, nodes_after_pooling, channels]
                - idx: Selection indices with shape [batch, nodes_before_pooling] as returned by TopKPool
                - A: Original adjacency matrix with shape [batch, nodes_before_pooling, nodes_before_pooling]
        
        Returns:
            Unpooled node features with shape [batch, nodes_before_pooling, channels]
        """
        x, idx, A = inputs
        
        # Get dimensions
        batch_size = tf.shape(x)[0]
        n_pooled_nodes = tf.shape(x)[1]  # Number of nodes after pooling
        n_features = tf.shape(x)[2]      # Number of features per node
        n_original_nodes = tf.shape(A)[1]  # Number of nodes before pooling
        
        # Create output tensor of zeros with shape [batch_size, n_original_nodes, n_features]
        x_out = tf.zeros([batch_size, n_original_nodes, n_features], dtype=x.dtype)
        
        # For each batch
        for b in range(batch_size):
            # Get the indices for this batch
            # idx[b] is a binary mask of shape [n_original_nodes]
            # We need to find which indices are True (1)
            indices = tf.where(idx[b])  # Shape [num_selected_nodes, 1]
            indices = tf.cast(tf.squeeze(indices, axis=1), dtype=tf.int32)  # Shape [num_selected_nodes]
            
            # Get features for this batch
            features = x[b]  # Shape [n_pooled_nodes, n_features]
            
            # Create batch indices
            batch_idx = tf.ones_like(indices, dtype=tf.int32) * b
            
            # Create scatter indices of shape [num_selected_nodes, 2]
            # Each row is [batch_idx, node_idx]
            scatter_indices = tf.stack([batch_idx, indices], axis=1)
            
            # Update x_out with the features at the right positions
            x_out = tf.tensor_scatter_nd_update(
                x_out,
                scatter_indices,
                features[:tf.shape(indices)[0]]  # Ensure we only take as many features as we have indices
            )
        
        return x_out


def gaussian_blur_3d(x, kernel_size=3, sigma=1.0):
    kernel = np.fromfunction(
        lambda x, y, z: (1/(2*np.pi*sigma**2)) * np.exp(-((x - kernel_size//2)**2 + (y - kernel_size//2)**2 + (z - kernel_size//2)**2) / (2*sigma**2)),
        (kernel_size, kernel_size, kernel_size)
    )
    kernel = kernel / np.sum(kernel)
    kernel = kernel[:, :, :, np.newaxis, np.newaxis]
    
    x = tf.nn.conv3d(x, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    
    return x[0, :, :, :, 0]
    
    
def train_models(models, losses, optimizers, data_generator, validation_data_generator, epochs=10, steps_per_epoch=100, batch_size=128, use_checkpoints=False, ckpt_dir='ckpts'):
    for model, loss, optimizer in zip(models, losses, optimizers):
        if type(model) == GraphVariationalAutoencoder or type(model) == BatchGraphVariationalAutoencoder:
            model.compile(optimizer=optimizer, loss=loss, run_eagerly=True)
        else:
            model.compile(optimizer=optimizer, loss=loss)
        model.build(next(iter(data_generator))[0].shape[1:])
        
        batch_x, batch_y = next(iter(data_generator))
        model.train_on_batch(batch_x, batch_y)
        
        model.summary()

        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, directory=f'{ckpt_dir}/saved_{model.name}_ckpt', max_to_keep=3)
        
        if ckpt_manager.latest_checkpoint and use_checkpoints:
            ckpt.restore(ckpt_manager.latest_checkpoint).assert_existing_objects_matched()
            print('restored from checkpoint:', ckpt_manager.latest_checkpoint)
        elif use_checkpoints:
            print('no checkpoint found--initializing from scratch.')

        checkpoint_callback = CheckpointCallback(ckpt_manager)
        callbacks = [checkpoint_callback] if use_checkpoints else []

        model.loss = loss        
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
        
        batch_x, batch_y = next(iter(data_generator))
        model.train_on_batch(batch_x, batch_y)
        
        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, directory=f'ckpts/saved_{model.name}_ckpt', max_to_keep=3)
        
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).assert_existing_objects_matched()
            print('restored from checkpoint:', ckpt_manager.latest_checkpoint)
        else:
            print('no checkpoint found--initializing from scratch.')
        
        model.summary()

class DiffUnpool(tf.keras.layers.Layer):
    """
    DiffUnpool layer for graph neural networks.
    This layer reverses the operation performed by DiffPool.
    
    Args:
        **kwargs: Additional arguments for the Layer class.
    """
    def __init__(self, **kwargs):
        super(DiffUnpool, self).__init__(**kwargs)
        self.supports_masking = False

    def build(self, input_shape):
        self.built = True

    @tf.function
    def call(self, inputs):
        """
        Implements the unpooling operation using a selection matrix.
        
        Args:
            inputs: List containing:
                - x: Pooled node features tensor with shape [batch, nodes_after_pooling, channels]
                - S: Selection matrix with shape [batch, nodes_before_pooling, nodes_after_pooling]
                   This is the transpose of the selection matrix used in pooling
                - A: Original adjacency matrix with shape [batch, nodes_before_pooling, nodes_before_pooling]
        
        Returns:
            Unpooled node features with shape [batch, nodes_before_pooling, channels]
        """
        x, S, A = inputs
        
        # Get dimensions
        batch_size = tf.shape(x)[0]
        n_pooled_nodes = tf.shape(x)[1]    # Number of nodes after pooling
        n_features = tf.shape(x)[2]        # Number of features per node
        n_original_nodes = tf.shape(S)[1]  # Number of nodes before pooling
        
        # Initialize output tensor
        x_out = tf.zeros([batch_size, n_original_nodes, n_features], dtype=x.dtype)
        
        # For each batch
        for b in range(batch_size):
            # Get selection matrix and features for this batch
            S_b = S[b]  # Shape [n_original_nodes, n_pooled_nodes]
            x_b = x[b]  # Shape [n_pooled_nodes, n_features]
            
            # Compute unpooled features by matrix multiplication:
            # S_b @ x_b produces a tensor of shape [n_original_nodes, n_features]
            # This distributes the pooled features back to the original nodes
            # based on the selection weights
            x_unpooled = tf.matmul(S_b, x_b)
            
            # Update the output tensor for this batch
            x_out = tf.tensor_scatter_nd_update(
                x_out,
                tf.stack([tf.ones(n_original_nodes, dtype=tf.int32) * b, 
                         tf.range(n_original_nodes, dtype=tf.int32)], axis=1),
                x_unpooled
            )
        
        return x_out

class BatchGraphVariationalAutoencoder(tf.keras.Model):
    def __init__(
        self, 
        input_shape, 
        latent_dim, 
        encoder_layer_sizes=[], 
        decoder_layer_sizes=[], 
        pooling=True,
        adjacency_matrix=None,
        name='batch_graph_variational_autoencoder'
    ):
        super().__init__(name=name)
        
        self.latent_dim = latent_dim
        self.pooling = pooling
        self.input_shape = input_shape
        self.encoder_layer_sizes = encoder_layer_sizes
        self.decoder_layer_sizes = decoder_layer_sizes

        # Preprocess adjacency
        # adjacency_matrix = localpooling_filter(adjacency_matrix, symmetric=True)
        # self.adjacency_matrix = tf.convert_to_tensor(adjacency_matrix.toarray(), dtype=tf.float32)
        tf.debugging.assert_all_finite(adjacency_matrix, "Adjacency fed to the model has NaNs/Infs")
        self.adjacency_matrix = tf.cast(adjacency_matrix, tf.float32)

        # Encoder layers
        self.encoder_layers = []
        self.pooling_layers = []
        
        # Add GCN + pooling
        nodes = input_shape[0]
        for units in encoder_layer_sizes:
            self.encoder_layers.append(spektral.layers.GCNConv(units, activation='relu'))
            if pooling:
                nodes //= 2
                self.pooling_layers.append(spektral.layers.DiffPool(k=nodes, return_selection=True))
        
        # Final encoder layer -> produce mean+log_var
        self.encoder_layers.append(spektral.layers.GCNConv(latent_dim * 2))
        
        # Decoder layers
        self.decoder_layers = []
        self.unpooling_layers = []
        
        # Initial decoder layer
        self.decoder_layers.append(spektral.layers.GCNConv(decoder_layer_sizes[0], activation='relu'))
        
        # Add GCN + unpooling
        for units in decoder_layer_sizes[1:]:
            if pooling:
                self.unpooling_layers.append(DiffUnpool())
            self.decoder_layers.append(spektral.layers.GCNConv(units, activation='relu'))
        
        # Final decoder layer
        self.decoder_layers.append(spektral.layers.GCNConv(input_shape[1], activation='softplus'))
        
        # Metric trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.pooling_loss_tracker = tf.keras.metrics.Mean(name="pooling_loss")

    def reparameterize(self, mean, log_var):
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * eps

    def encode(self, x, training=False):
        """
        Encode the entire batch at once.
        Returns mean, log_var, and tracking information for the decode step.
        
        Args:
            x: Input tensor with shape [batch_size, n_nodes, n_features]
            training: Training mode flag
            
        Returns:
            mean: Mean of the latent distribution
            log_var: Log variance of the latent distribution
            adjacency_stack: List of adjacency matrices for each pooling level
            selection_stack: List of selection matrices from pooling operations
        """
        batch_size = tf.shape(x)[0]
        
        # Create batch version of adjacency matrix
        batch_adj = tf.repeat(tf.expand_dims(self.adjacency_matrix, axis=0), batch_size, axis=0)
        
        # Track adjacency and selection matrices
        adjacency_stack = [batch_adj]
        selection_stack = []
        
        # Current feature tensor
        x_enc = x
        current_adj = batch_adj
        
        # Process through encoder layers
        for i, layer in enumerate(self.encoder_layers[:-1]):
            # Apply GCN convolution
            print('Encoder layer:', i, f'{layer.name}:', x_enc.shape, current_adj.shape)
            tf.debugging.assert_all_finite(x_enc, "x_enc (input) has NaNs/Infs")
            tf.debugging.assert_all_finite(current_adj, "A (input) has NaNs/Infs")

            assert self.adjacency_matrix.dtype == tf.float32
            assert tf.reduce_min(tf.reduce_sum(self.adjacency_matrix, -1)) > 0

            x_enc = layer([x_enc, current_adj], training=training)

            tf.debugging.assert_all_finite(x_enc, "x_enc (gcn output) has NaNs/Infs")
            
            # Apply pooling if needed
            if self.pooling and i < len(self.pooling_layers):
                pool_layer = self.pooling_layers[i]
                
                # Process each sample in the batch
                pooled_features = []
                pooled_adj = []
                selection_indices = []
                
                for b in range(batch_size):
                    # Convert to sparse for pooling
                    adj_sparse = tf.sparse.from_dense(current_adj[b])
                    # pool_layer._n_nodes = tf.shape(current_adj[b])[0]

                    tf.debugging.assert_all_finite(x_enc[b], "x_enc_b (input) has NaNs/Infs")
                    tf.debugging.assert_all_finite(tf.sparse.to_dense(adj_sparse), "adj_sparse_b has NaNs/Infs")
                    
                    # Apply pooling
                    features_b, adj_b, selection_b = pool_layer(
                        [x_enc[b], adj_sparse], 
                        training=training
                    )

                    tf.debugging.assert_all_finite(features_b, "features_b has NaNs/Infs")
                    tf.debugging.assert_all_finite(adj_b, "adj_b has NaNs/Infs")
                    tf.debugging.assert_all_finite(selection_b, "selection_b has NaNs/Infs")

                    adj_b = tf.where(tf.math.is_finite(adj_b), adj_b, 0.0)
                    
                    pooled_features.append(features_b)
                    pooled_adj.append(adj_b)
                    selection_indices.append(selection_b)
                
                # Stack results
                x_enc = tf.stack(pooled_features, axis=0)
                current_adj = tf.stack(pooled_adj, axis=0)
                selection_stack.append(tf.stack(selection_indices, axis=0))
                adjacency_stack.append(current_adj)
        
        # Final encoder layer
        x_enc = self.encoder_layers[-1]([x_enc, current_adj], training=training)
        
        # Split into mean and log_var
        print('x_enc:', x_enc.shape)
        mean, log_var = tf.split(x_enc, num_or_size_splits=2, axis=-1)

        print('mean:', mean.shape)
        print('log_var:', log_var.shape)
        
        return mean, log_var, adjacency_stack, selection_stack

    def decode(self, z, adjacency_stack, selection_stack, training=False):
        """
        Decode the entire batch at once.
        
        Args:
            z: Latent vector with shape [batch_size, n_nodes_pooled, latent_dim]
            adjacency_stack: List of adjacency matrices from encoding
            selection_stack: List of selection matrices from encoding
            training: Training mode flag
            
        Returns:
            Reconstructed output with shape [batch_size, n_nodes, n_features]
        """
        batch_size = tf.shape(z)[0]
        
        # Start with the adjacency from the last encode step
        current_adj = adjacency_stack[-1]
        
        # Iterate through decoder layers
        num_unpools = len(self.unpooling_layers) 

        x_dec = z
        
        for i, gcn_layer in enumerate(self.decoder_layers[:-1]):
            print('Decoder layer:', i, f'{gcn_layer.name}:', x_dec.shape, current_adj.shape, 'layer size:', self.decoder_layer_sizes[i])
            # Apply GCN layer
            x_dec = gcn_layer([x_dec, current_adj], training=training)

            # Apply unpooling if applicable
            if self.pooling and i < num_unpools:
                unpool_layer = self.unpooling_layers[num_unpools - 1 - i]
                
                # Calculate indices for adjacency and selection
                adj_idx = len(adjacency_stack) - 2 - i
                sel_idx = len(selection_stack) - 1 - i
                
                if adj_idx >= 0 and sel_idx >= 0:
                    # Get previous adjacency and selection matrices
                    old_adj = adjacency_stack[adj_idx]
                    selection = selection_stack[sel_idx]
                    
                    # Apply unpooling for each sample
                    unpooled_features = []
                    for b in range(batch_size):
                        unpooled_b = unpool_layer([
                            tf.expand_dims(x_dec[b], 0),
                            tf.expand_dims(selection[b], 0),
                            tf.expand_dims(old_adj[b], 0)
                        ])
                        unpooled_features.append(tf.squeeze(unpooled_b, axis=0))
                    
                    # Stack results
                    x_dec = tf.stack(unpooled_features, axis=0)
                    current_adj = tf.cast(old_adj, x_dec.dtype)
        
        # Apply final decoder layer
        current_adj = adjacency_stack[0]
        print('Final decoder layer:', f'{self.decoder_layers[-1].name}:', x_dec.shape, current_adj.shape)
        x_dec = self.decoder_layers[-1]([x_dec, current_adj], training=training)
        
        return x_dec

    def call(self, x, training=False):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor with shape [batch_size, n_nodes, n_features]
            training: Training mode flag
            
        Returns:
            reconstructed: Reconstructed output
            mean: Mean of the latent distribution
            log_var: Log variance of the latent distribution
        """
        # Encode input
        mean, log_var, adjacency_stack, selection_stack = self.encode(x, training=training)
        
        # Sample from latent distribution
        z = self.reparameterize(mean, log_var)

        print('z:', z.shape)
        
        # Decode latent representation
        reconstructed = self.decode(z, adjacency_stack, selection_stack, training=training)
        
        return reconstructed, mean, log_var

    def train_step(self, data):
        x = data[0]
        x = tf.cast(x, tf.float32)
        
        with tf.GradientTape() as tape:
            reconstructed, mean, log_var = self(x, training=True)
            
            # Compute losses
            r_loss = reconstruction_loss(x, reconstructed)  
            kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
            
            loss = r_loss + kl_loss * 0.05
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(r_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            'loss': self.loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        x = data[0]
        x = tf.cast(x, tf.float32)
        reconstructed, mean, log_var = self(x, training=False)
        r_loss = reconstruction_loss(x, reconstructed)
        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
        
        # Calculate total loss
        loss = r_loss + kl_loss * 0.05
        
        # Update metrics
        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(r_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            'loss': self.loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result()
        }

    @property
    def metrics(self):
        metrics = [self.loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]
        if self.pooling:
            metrics.append(self.pooling_loss_tracker)
        return metrics
        
    def encode_latent(self, x):
        """
        Encode input to latent representation for downstream tasks.
        
        Args:
            x: Input tensor with shape [batch_size, n_nodes, n_features]
            
        Returns:
            Latent representation
        """
        mean, log_var, _, _ = self.encode(x, training=False)
        return self.reparameterize(mean, log_var)
        
    def summary(self):
        print(f'\n\033[1mModel: {self.name}\033[0m')
        print("\nEncoder:")
        print(f'{"Layer":<30} {"Output Shape":<20} {"Params":<10}')
        print('-' * 60)
        
        for i, layer in enumerate(self.encoder_layers):
            print(f'{layer.name:<30} {"?":<20} {layer.count_params():<10}')
            if i < len(self.pooling_layers) and self.pooling:
                print(f'{self.pooling_layers[i].name:<30} {"?":<20} {self.pooling_layers[i].count_params():<10}')
        
        print("\nDecoder:")
        print(f'{"Layer":<30} {"Output Shape":<20} {"Params":<10}')
        print('-' * 60)
        
        for i, layer in enumerate(self.decoder_layers):
            print(f'{layer.name:<30} {"?":<20} {layer.count_params():<10}')
            if i < len(self.unpooling_layers) and self.pooling:
                print(f'{self.unpooling_layers[i].name:<30} {"?":<20} {0:<10}')