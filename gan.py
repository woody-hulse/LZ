import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions
import spektral
from spektral.utils import normalized_laplacian, rescale_laplacian
import numpy as np

import re
import os
import seaborn as sns
from scipy import stats
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
import multiprocessing as mp
from functools import partial

from preprocessing import *
from generator import *
from simple_pulse import Params

from autoencoder_analysis import plot_3d_scatter_with_profiles


class Generator(tf.keras.Model):
    def __init__(
        self,
        output_shape,
        latent_dim,
        layer_sizes=[],
        name='dense_generator'
    ):
        super().__init__(name=name)
        
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.layer_sizes = layer_sizes
        
        self.layers_list = []

        for i, units in enumerate(layer_sizes):
            self.layers_list.append(tf.keras.layers.Dense(units, activation='sigmoid', name=f'generator_dense_{i}'))
        
        self.layers_list.append(tf.keras.layers.Dense(np.prod(output_shape), activation='linear', name='generator_output'))
        self.layers_list.append(tf.keras.layers.Reshape(output_shape, name='generator_reshape'))
    
    def call(self, y, training=False):
        z = tf.random.uniform(shape=tf.shape(y))
        x = tf.concat([y, z], axis=-1)
        for layer in self.layers_list:
            x = layer(x)
        
        return x
    
    def summary(self):
        print(f'\n\033[1mModel: {self.name}\033[0m')
        print(f'{"Layer":<30} {"Output Shape":<20} {"Params":<10}')
        print('-' * 60)
        
        # Print layers
        for layer in self.layers_list:
            print(f'{layer.name:<30} {"?":<20} {layer.count_params():<10}')
        print()

import tensorflow as tf


def _upsample_3d(x):
    """Nearest-neighbour ×2 in z, y, and x, then scale by ½."""
    x = tf.repeat(x, 2, axis=1)        # depth (z)
    x = tf.repeat(x, 2, axis=2)        # height (y)
    x = tf.repeat(x, 2, axis=3)        # width  (x)
    return x * 0.5                     # √2-normalisation per axis (empirical)


def inverse_haar3d(coeffs):
    """
    coeffs  :  (B, D/2, H/2, W/2, 8·C)
               channel layout  ⟨LLL‖LLH‖LHL‖LHH‖HLL‖HLH‖HHL‖HHH⟩
    returns :  (B, D,   H,   W,   C)
    """
    # ――― 1. split sub-bands -------------------------------------------------
    (LLL, LLH, LHL, LHH,
     HLL, HLH, HHL, HHH) = tf.split(coeffs, 8, axis=-1)

    # ――― 2. upsample every band to full resolution --------------------------
    LLL = _upsample_3d(LLL)
    LLH = _upsample_3d(LLH)
    LHL = _upsample_3d(LHL)
    LHH = _upsample_3d(LHH)
    HLL = _upsample_3d(HLL)
    HLH = _upsample_3d(HLH)
    HHL = _upsample_3d(HHL)
    HHH = _upsample_3d(HHH)

    # ――― 3. parity masks (±1) along each spatial axis -----------------------
    D, H, W = tf.shape(LLL)[1], tf.shape(LLL)[2], tf.shape(LLL)[3]
    z_mask = tf.reshape(1.0 - 2.0 * tf.cast(tf.range(D) % 2, tf.float32),
                        [1, D, 1, 1, 1])
    y_mask = tf.reshape(1.0 - 2.0 * tf.cast(tf.range(H) % 2, tf.float32),
                        [1, 1, H, 1, 1])
    x_mask = tf.reshape(1.0 - 2.0 * tf.cast(tf.range(W) % 2, tf.float32),
                        [1, 1, 1, W, 1])

    # ――― 4. recombine --------------------------------------------------------
    recon = (
        LLL                                                   +
        LLH *                x_mask                           +
        LHL *        y_mask                                   +
        LHH *        y_mask * x_mask                          +
        HLL * z_mask                                          +
        HLH * z_mask *                x_mask                  +
        HHL * z_mask * y_mask                                 +
        HHH * z_mask * y_mask * x_mask
    )
    # (optional) divide by 4 or 8 if you need exact orthonormal energy;
    # leaving as-is lets the network learn the overall gain.
    return recon



def _upsample_3d(x):
    """Nearest-neighbour ×2 in z, y, and x, then scale by ½."""
    x = tf.repeat(x, 2, axis=1)        # depth (z)
    x = tf.repeat(x, 2, axis=2)        # height (y)
    x = tf.repeat(x, 2, axis=3)        # width  (x)
    return x * 0.5                     # √2-normalisation per axis (empirical)


def inverse_haar3d(coeffs):
    """
    coeffs  :  (B, D/2, H/2, W/2, 8·C)
               channel layout  ⟨LLL‖LLH‖LHL‖LHH‖HLL‖HLH‖HHL‖HHH⟩
    returns :  (B, D,   H,   W,   C)
    """
    # ――― 1. split sub-bands -------------------------------------------------
    (LLL, LLH, LHL, LHH,
     HLL, HLH, HHL, HHH) = tf.split(coeffs, 8, axis=-1)

    # ――― 2. upsample every band to full resolution --------------------------
    LLL = _upsample_3d(LLL)
    LLH = _upsample_3d(LLH)
    LHL = _upsample_3d(LHL)
    LHH = _upsample_3d(LHH)
    HLL = _upsample_3d(HLL)
    HLH = _upsample_3d(HLH)
    HHL = _upsample_3d(HHL)
    HHH = _upsample_3d(HHH)

    # ――― 3. parity masks (±1) along each spatial axis -----------------------
    D, H, W = tf.shape(LLL)[1], tf.shape(LLL)[2], tf.shape(LLL)[3]
    z_mask = tf.reshape(1.0 - 2.0 * tf.cast(tf.range(D) % 2, tf.float32),
                        [1, D, 1, 1, 1])
    y_mask = tf.reshape(1.0 - 2.0 * tf.cast(tf.range(H) % 2, tf.float32),
                        [1, 1, H, 1, 1])
    x_mask = tf.reshape(1.0 - 2.0 * tf.cast(tf.range(W) % 2, tf.float32),
                        [1, 1, 1, W, 1])

    # ――― 4. recombine --------------------------------------------------------
    recon = (
        LLL                                                   +
        LLH *                x_mask                           +
        LHL *        y_mask                                   +
        LHH *        y_mask * x_mask                          +
        HLL * z_mask                                          +
        HLH * z_mask *                x_mask                  +
        HHL * z_mask * y_mask                                 +
        HHH * z_mask * y_mask * x_mask
    )
    # (optional) divide by 4 or 8 if you need exact orthonormal energy;
    # leaving as-is lets the network learn the overall gain.
    return recon

# ---------- model -----------------------------------------------------------
class WaveletGenerator(tf.keras.Model):
    """
    Predicts wavelet coefficients and reconstructs an image / volume
    with a (inverse) Haar wavelet, fully differentiable.
    """
    def __init__(
            self,
            output_shape,          # e.g. (H, W, C) or (D, H, W, C)
            latent_dim,
            layer_sizes=None,
            name="wavelet_generator"):
        super().__init__(name=name)
        if layer_sizes is None:
            layer_sizes = [256, 512, 1024]

        self.latent_dim     = latent_dim
        self.output_shape   = output_shape
        self.coeffs_shape   = (output_shape[0] // 2, output_shape[1] // 2, output_shape[2] // 2, 8)
        self.output_shape   = (output_shape[0] * output_shape[1], output_shape[2])
        self.dims           = len(output_shape) - 1   # channels excluded

        # ------------------------------------------------------------------
        # Dense trunk
        # ------------------------------------------------------------------
        self.layers_list = []
        for i, units in enumerate(layer_sizes):
            self.layers_list.append(
                tf.keras.layers.Dense(units, activation="relu",
                                      name=f"wavelet_dense_{i}")
            )

        # final layer: produce wavelet-coeff tensor
        # number of coeffs == np.prod(output_shape) (1-level Haar)
        coeff_dim = int(np.prod(self.coeffs_shape))
        self.layers_list.append(
            tf.keras.layers.Dense(coeff_dim, activation="linear",
                                  name="wavelet_coeff_flat")
        )
        self.reshape_coeffs = tf.keras.layers.Reshape(self.coeffs_shape,
                                               name="coeff_reshape")
        self.reshape_output = tf.keras.layers.Reshape(self.output_shape,
                                               name="output_reshape")

    # ----------------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------------
    def call(self, y, training=False):
        """
        y: conditioning tensor (batch, cond_dim)
        latent z is drawn per sample.
        """
        z = tf.random.normal(shape=(tf.shape(y)[0], self.latent_dim))
        x = tf.concat([y, z], axis=-1)
        for layer in self.layers_list:
            x = layer(x)
        coeffs = self.reshape_coeffs(x)              # (B, *out_shape)

        # differentiable inverse wavelet
        recon = inverse_haar3d(coeffs)
        recon = self.reshape_output(recon)
        return recon


class DistributionGenerator(tf.keras.Model):
    def __init__(self, output_shape, latent_dim, layer_sizes=[256, 512, 1024], name="distribution_generator"):
        super().__init__(name=name)

        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.layer_sizes = layer_sizes
        
        self.layers_list = []
        for i, units in enumerate(layer_sizes):
            self.layers_list.append(tf.keras.layers.Dense(units, activation='relu', name=f'generator_dense_{i}'))
        
        self.mu_output = [
            tf.keras.layers.Dense(np.prod(output_shape), activation='linear', name='generator_mu_output'),
            tf.keras.layers.Reshape(output_shape, name='generator_mu_reshape')
        ]

        self.sigma_output = [
            tf.keras.layers.Dense(np.prod(output_shape), activation='softplus', name='generator_sigma_output'),
            tf.keras.layers.Reshape(output_shape, name='generator_sigma_reshape')
        ]
        
    def call(self, y, training=False):
        for layer in self.layers_list:
            y = layer(y)
        
        mu = y
        for layer in self.mu_output:
            mu = layer(mu)

        sigma = y
        for layer in self.sigma_output:
            sigma = layer(sigma)

        reparametrize = mu + sigma * tf.random.normal(shape=tf.shape(mu))
        reparametrize_abs = tf.abs(reparametrize)
        return reparametrize
        

class NormalTransposeConvLayer(tf.keras.layers.Layer):
    def __init__(self, input_shape, filter_size, strides, name="normal_transpose_conv_layer"):
        super().__init__(name=name)
        self.input_shape = input_shape
        self.filter_size = filter_size
        self.strides = strides
        
        self.output_shape = (self.input_shape[0] * self.strides[0], self.input_shape[1] * self.strides[1], self.input_shape[2] * self.strides[2], 1)
        self.transpose_conv_layer = tf.keras.layers.Conv3DTranspose(filters=1, kernel_size=self.filter_size, strides=self.strides, padding='same', activation='relu')
        
        self.mu_layer = tf.keras.layers.Dense(np.prod(self.output_shape), activation='linear', name='generator_mu_output')
        self.sigma_layer = tf.keras.layers.Dense(np.prod(self.output_shape), activation='softplus', name='generator_sigma_output')
        self.reshape_layer = tf.keras.layers.Reshape(self.output_shape, name='generator_reshape')

    def call(self, inputs):
        transpose_conv = self.transpose_conv_layer(inputs)
        mu = self.mu_layer(inputs)
        sigma = self.sigma_layer(inputs)

        reparametrize = mu + sigma * tf.random.normal(shape=tf.shape(mu))
        reparametrize_abs = tf.abs(reparametrize) # maybe change to relu

        return transpose_conv + reparametrize_abs

    def build(self, input_shape):
        super().build(input_shape)

class ConvDistributionGenerator(tf.keras.Model):
    def __init__(self, output_shape, latent_dim, layer_sizes=[256, 512, 1024], name="conv_distribution_generator"):
        super().__init__(name=name)

        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.layer_sizes = layer_sizes

        self.tail_layers = []
        for i, units in enumerate(layer_sizes):
            self.tail_layers.append(tf.keras.layers.Dense(units, activation='relu', name=f'generator_dense_{i}'))

        self.conv_filters = [32, 16, 1]
        self.num_conv_layers = len(self.conv_filters)
        self.stride = (1, 1, 1)
        self.filter_size = (3, 3, 3)

        conv_input_shape = (self.output_shape[0] // (self.stride[0]**self.num_conv_layers), self.output_shape[1] // (self.stride[1]**self.num_conv_layers), self.output_shape[2] // (self.stride[2]**self.num_conv_layers), 1)

        self.head_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(np.prod(conv_input_shape), activation='relu', name='generator_conv_input_flat'),
            tf.keras.layers.Reshape(conv_input_shape, name='generator_conv_input_reshape')
        ])

        self.head_blocks = []
        for i, filters in enumerate(self.conv_filters):
            transpose_conv_layer = tf.keras.layers.Conv3DTranspose(filters=filters, kernel_size=self.filter_size, strides=self.stride, padding='same', activation='relu')
            conv_input_shape = (self.output_shape[0] // (self.stride[0]**(i+1)), self.output_shape[1] // (self.stride[1]**(i+1)), self.output_shape[2] // (self.stride[2]**(i+1)), 1)

            mu_layer = tf.keras.layers.Dense(np.prod(conv_input_shape), activation='linear', name=f'generator_mu_output_{i}')
            sigma_layer = tf.keras.layers.Dense(np.prod(conv_input_shape), activation='softplus', name=f'generator_sigma_output_{i}')
            reshape_layer = tf.keras.layers.Reshape(conv_input_shape, name=f'generator_reshape_{i}')

            self.head_blocks.append([
                transpose_conv_layer,
                mu_layer,
                sigma_layer,
                reshape_layer
            ])

        self.reshape_layer = tf.keras.layers.Reshape((self.output_shape[0] * self.output_shape[1], self.output_shape[2]), name='generator_reshape')
    
    def call(self, y, training=False):
        for layer in self.tail_layers:
            y = layer(y)

        base_latent = y
        output = self.head_layers(base_latent)

        for transpose_conv_layer, mu_layer, sigma_layer, reshape_layer in self.head_blocks:
            output = transpose_conv_layer(output)
            mu = mu_layer(base_latent)
            sigma = sigma_layer(base_latent)

            reparametrize = mu + sigma * tf.random.normal(shape=tf.shape(mu))
            # reparametrize_abs = tf.abs(reparametrize) # maybe change to relu
            reparametrize_relu = tf.nn.relu(reparametrize)
            reparametrize_relu = reshape_layer(reparametrize_relu)

            output += reparametrize_relu

        return self.reshape_layer(output)
                

class Autoencoder(tf.keras.Model):
    def __init__(
        self,
        input_shape,
        latent_dim,
        encoder_layer_sizes=[],
        decoder_layer_sizes=[],
        name='dense_autoencoder'
    ):
        super().__init__(name=name)
        
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.encoder_layer_sizes = encoder_layer_sizes
        self.decoder_layer_sizes = decoder_layer_sizes
        
        # Create encoder
        self.encoder_layers = [
            tf.keras.layers.Flatten(),
        ]
        
        # Add encoder dense layers
        for i, units in enumerate(encoder_layer_sizes):
            self.encoder_layers.append(tf.keras.layers.Dense(units, activation='relu', name=f'encoder_dense_{i}'))
        
        # Output layer for latent space
        self.encoder_layers.append(tf.keras.layers.Dense(latent_dim, name='encoder_output'))
        
        # Create decoder
        self.decoder_layers = []
        
        # Add decoder dense layers
        for i, units in enumerate(decoder_layer_sizes):
            self.decoder_layers.append(tf.keras.layers.Dense(units, activation='relu', name=f'decoder_dense_{i}'))
        
        # Output layer for reconstruction
        self.decoder_layers.append(tf.keras.layers.Dense(np.prod(input_shape), activation='softplus', name='decoder_output'))
        self.decoder_layers.append(tf.keras.layers.Reshape(input_shape))
        
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
    
    def encode(self, x):
        # Process through encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        return x
    
    def decode(self, z):
        # If z has more than 2 dimensions, flatten it
        if len(z.shape) > 2:
            z = tf.reshape(z, [-1, z.shape[-1]])
        
        # Process through decoder layers
        for layer in self.decoder_layers:
            z = layer(z)
        
        return z
    
    def call(self, inputs, training=False):
        z = self.encode(inputs)
        reconstructed = self.decode(z)
        return reconstructed
    
    def summary(self):
        print(f'\n\033[1mModel: {self.name}\033[0m')
        print("\nEncoder:")
        print(f'{"Layer":<30} {"Output Shape":<20} {"Params":<10}')
        print('-' * 60)
        
        # Print encoder layers
        for layer in self.encoder_layers:
            print(f'{layer.name:<30} {"?":<20} {layer.count_params():<10}')
        
        print("\nDecoder:")
        print(f'{"Layer":<30} {"Output Shape":<20} {"Params":<10}')
        print('-' * 60)
        
        # Print decoder layers
        for layer in self.decoder_layers:
            print(f'{layer.name:<30} {"?":<20} {layer.count_params():<10}')
        
        print()
        
    def train_step(self, data):
        x = data[0]
        x = tf.cast(x, tf.float32)

        with tf.GradientTape() as tape:
            reconstructed = self(x, training=True)
            reconstruction_loss = tf.reduce_mean(tf.square(x - reconstructed))
            loss = reconstruction_loss
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.loss_tracker.update_state(loss)
        
        return {
            'loss': self.loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
        }
    
    def test_step(self, data):
        x = data[0]
        x = tf.cast(x, tf.float32)
        reconstructed = self(x, training=False)
        reconstruction_loss = tf.reduce_mean(tf.square(x - reconstructed))
        
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.loss_tracker.update_state(reconstruction_loss)

        return {
            'loss': self.loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
        }
        
    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.reconstruction_loss_tracker,
        ]
    
    def build(self, input_shape):
        super().build(input_shape)


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

#############################################
# 1) Discriminator with feature extraction   #
#############################################
class Discriminator(tf.keras.Model):
    """Dense conditional critic that can also return intermediate features
    suitable for the Salimans et al. (2016) *feature‑matching* loss."""

    def __init__(
        self,
        input_shape,
        layer_sizes=None,
        name="dense_critic",
    ):
        super().__init__(name=name)
        if layer_sizes is None:
            layer_sizes = [256, 128]

        self.input_shape_ = input_shape
        self.layer_sizes = layer_sizes

        # --- image branch (x) ---
        self.main_layers = [tf.keras.layers.Flatten(input_shape=input_shape)]
        for units in layer_sizes:
            self.main_layers.append(tf.keras.layers.Dense(units, activation="relu"))

        # --- condition branch (y) ---
        self.condition_layers = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
        ]

        # --- joint representation ---
        self.joint_layers = [
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
        ]

        # Minibatch discrimination layer
        self.minibatch_discrimination = MinibatchDiscrimination(32, 16)

        # Final linear layer for WGAN score
        self.final_dense = tf.keras.layers.Dense(1)

    # ------------------------------------------------------------------
    # Private helper that runs **everything up to** but excluding final_dense
    # ------------------------------------------------------------------
    def _forward_base(self, x, condition):
        x = tf.cast(x, tf.float32)
        condition = tf.cast(condition, tf.float32)

        # add a small amount of noise to x
        x = x + tf.random.normal(tf.shape(x), stddev=0.01)

        # image path
        for layer in self.main_layers:
            x = layer(x)

        # condition path
        for layer in self.condition_layers:
            condition = layer(condition)

        h = tf.concat([x, condition], axis=-1)
        for layer in self.joint_layers:
            h = layer(h)
        return h  # this is the **feature representation**

    # ------------------------------------------------------------------
    # Standard `call` – returns *only* the critic score (scalar per sample)
    # ------------------------------------------------------------------
    def call(self, inputs, training=False):
        x, condition = inputs
        features = self._forward_base(x, condition)
        
        # Apply minibatch discrimination
        batch_features = self.minibatch_discrimination(features)
        
        # Concatenate with original features
        enhanced_features = tf.concat([features, batch_features], axis=-1)
        
        # Final score
        score = self.final_dense(enhanced_features)
        return score

    # ------------------------------------------------------------------
    # Extra utility to obtain features without re‑computing the final score
    # ------------------------------------------------------------------
    def extract_features(self, inputs):
        x, condition = inputs
        return self._forward_base(x, condition)


class Conv3DDiscriminator(tf.keras.Model):
    """
    3-D convolutional WGAN-GP critic with optional feature extraction,
    accepting input grids of shape (batch, 24, 24, 700) and an
    arbitrary conditioning vector.
    """

    def __init__(
        self,
        input_shape,                    # (24, 24, 700)
        name="conv3d_critic",
    ):
        super().__init__(name=name)

        # ----- image branch -----
        self.image_layers = [
            tf.keras.layers.Reshape((*input_shape, 1)),      # (24,24,700,1)

            tf.keras.layers.Conv3D(128,  kernel_size=(3,3,13), strides=(1,1,4),
                                   padding="same", activation="relu"),
            tf.keras.layers.Conv3D(256, kernel_size=(3,3,7), strides=(1,1,2),
                                   padding="same", activation="relu"),
            tf.keras.layers.Conv3D(256, kernel_size=(3,3,3), strides=(1,1,1),
                                   padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling3D(),        # -> (256,)
        ]

        # ----- dense branch -----
        self.dense_layers = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
        ]

        # ----- condition branch -----
        self.condition_layers = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
        ]

        # ----- joint dense head -----
        self.joint_layers = [
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
        ]

        # Mini-batch discrimination (identical to your original utility)
        self.minibatch_discrimination = MinibatchDiscrimination(32, 16)

        # Final linear score for WGAN
        self.final_dense = tf.keras.layers.Dense(1)

    # ---------------------------------------------------------------
    # private backbone -> feature vector (before minibatch discrim)
    # ---------------------------------------------------------------
    def _forward_base(self, x, condition, training=False):
        x  = tf.cast(x, tf.float32)
        cond = tf.cast(condition, tf.float32)

        dense_x = x

        for layer in self.image_layers:
            x = layer(x, training=training) if hasattr(layer, "training") else layer(x)

        for layer in self.dense_layers:
            dense_x = layer(dense_x, training=training) if hasattr(layer, "training") else layer(dense_x)

        for layer in self.condition_layers:
            cond = layer(cond, training=training) if hasattr(layer, "training") else layer(cond)

        h = tf.concat([x, cond, dense_x], axis=-1)
        for layer in self.joint_layers:
            h = layer(h, training=training) if hasattr(layer, "training") else layer(h)
        return h  # feature vector (batch, 32)

    # ---------------------------------------------------------------
    # standard call – returns critic score (scalar per sample)
    # ---------------------------------------------------------------
    def call(self, inputs, training=False):
        x, condition = inputs
        features = self._forward_base(x, condition, training=training)

        # minibatch discrimination
        mb_feats = self.minibatch_discrimination(features)
        enhanced = tf.concat([features, mb_feats], axis=-1)

        return self.final_dense(enhanced)   # (batch, 1)

    # ---------------------------------------------------------------
    # handy hook for feature-matching loss
    # ---------------------------------------------------------------
    def extract_features(self, inputs, training=False):
        x, condition = inputs
        return self._forward_base(x, condition, training=training)


#################################################
# 2) GANTrainer with *feature‑matching* loss     #
#################################################
class GANTrainer:
    """GAN training loop with optional feature‑matching regularisation.

    Parameters
    ----------
    autoencoder : `tf.keras.Model`
        A trained encoder (you pass only the *encoder* part in practice).
    generator : `tf.keras.Model`
        Conditional generator network.
    discriminator : `Discriminator`
        Critic that supports `.extract_features()`.
    feature_matching_weight : float, default 10.0
        λ in     L_G = w_g * L_GAN  +  λ * L_fm
        Set 0 to disable feature matching.
    """

    def __init__(
        self,
        autoencoder,
        generator,
        discriminator,
        *,
        reconstruction_weight=0.0,
        generator_weight=1.0,
        feature_matching_weight=10.0,
        critic_steps=5,
        gradient_penalty_weight=10.0,
    ):
        self.autoencoder = autoencoder
        self.generator = generator
        self.discriminator = discriminator

        self.reconstruction_weight = reconstruction_weight
        self.generator_weight = generator_weight
        self.feature_matching_weight = feature_matching_weight

        self.critic_steps = critic_steps
        self.gradient_penalty_weight = gradient_penalty_weight

        # Optimisers
        self.ae_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.gen_optimizer = tf.keras.optimizers.RMSprop(5e-5)
        self.disc_optimizer = tf.keras.optimizers.RMSprop(5e-5)

        # Metrics (minimal set – add more if needed)
        self.metric_fm = tf.keras.metrics.Mean(name="feature_matching_loss")
        self.metric_gen = tf.keras.metrics.Mean(name="generator_loss")
        self.metric_disc = tf.keras.metrics.Mean(name="critic_loss")
        self.metric_critic_accuracy = tf.keras.metrics.Mean(name="critic_accuracy")
        self.metric_generator_fool_rate = tf.keras.metrics.Mean(name="generator_fool_rate")
        self.metric_wasserstein = tf.keras.metrics.Mean(name="wasserstein_distance")
        self.metric_gp = tf.keras.metrics.Mean(name="gradient_penalty")
        self.metric_recon = tf.keras.metrics.Mean(name="reconstruction_loss")

    # =======================================================
    # ===       Private helpers (gradient penalty)        ===
    # =======================================================
    def _gradient_penalty(self, real_x, fake_x, cond_y):
        batch = tf.shape(real_x)[0]
        alpha = tf.random.uniform([batch, 1, 1], 0.0, 1.0)
        inter = real_x + alpha * (fake_x - real_x)

        with tf.GradientTape() as tape:
            tape.watch(inter)
            score_inter = self.discriminator([inter, cond_y], training=True)
        grads = tape.gradient(score_inter, inter)
        grads_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean(tf.square(grads_norm - 1.0))
        return gp

    # =======================================================
    # ===                Train‑step (tf.function)        ===
    # =======================================================
    @tf.function
    def train_step(self, real_x, cond_y):
        real_x = tf.cast(real_x, tf.float32)
        cond_y = tf.cast(cond_y, tf.float32)

        # ‑‑ Get latent code from *frozen* encoder ‑‑
        z = self.autoencoder.encode(real_x)  # shape: [B, latent_dim]

        # ===============   1) Update Critic   ==================
        for _ in range(self.critic_steps):
            with tf.GradientTape() as disc_tape:
                fake_x = self.generator(z, training=True)
                real_score = self.discriminator([real_x, cond_y], training=True)
                fake_score = self.discriminator([fake_x, cond_y], training=True)

                wasserstein = tf.reduce_mean(fake_score - real_score)
                gp = self._gradient_penalty(real_x, fake_x, cond_y)
                disc_loss = wasserstein + self.gradient_penalty_weight * gp

            disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

        # ===============   2) Update Generator  ===============
        with tf.GradientTape() as gen_tape:
            fake_x = self.generator(z, training=True)
            reconstruction_loss = tf.reduce_mean(tf.square(real_x - fake_x))
            self.metric_recon.update_state(reconstruction_loss)
            
            fake_score = self.discriminator([fake_x, cond_y], training=False)
            gen_loss_gan = -tf.reduce_mean(fake_score)  # WGAN generator loss

            # --------‑‑ Feature‑matching term ‑‑---------
            fm_loss = 0.0
            if self.feature_matching_weight > 0.0:
                real_feat = self.discriminator.extract_features([real_x, cond_y])
                fake_feat = self.discriminator.extract_features([fake_x, cond_y])
                # mean over batch, then L2 over units
                fm_loss = tf.reduce_mean(
                    tf.square(tf.reduce_mean(real_feat, axis=0) - tf.reduce_mean(fake_feat, axis=0))
                )
                self.metric_fm.update_state(fm_loss)

            gen_loss = (
                self.generator_weight * gen_loss_gan
                + self.feature_matching_weight * fm_loss
            )

        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))

        # Calculate critic accuracy (how often real > fake)
        # We want critic to correctly assign higher scores to real data
        real_score_final = self.discriminator([real_x, cond_y], training=False)
        fake_score_final = self.discriminator([fake_x, cond_y], training=False)
        critic_accuracy = tf.reduce_mean(tf.cast(real_score_final > fake_score_final, tf.float32))
        
        # Calculate generator fool rate (how often fake score > 0)
        # In WGAN, discriminator tries to make real scores high and fake scores low
        # Generator tries to make fake scores high
        generator_fool_rate = tf.reduce_mean(tf.cast(fake_score_final > 0, tf.float32))
        
        # Calculate Wasserstein distance for monitoring
        wasserstein_distance = tf.reduce_mean(real_score_final - fake_score_final)

        # ===== Log/return losses =====
        self.metric_gen.update_state(gen_loss_gan)
        self.metric_disc.update_state(disc_loss)
        self.metric_critic_accuracy.update_state(critic_accuracy)
        self.metric_generator_fool_rate.update_state(generator_fool_rate)
        self.metric_wasserstein.update_state(wasserstein_distance)
        self.metric_gp.update_state(gp)

        return {
            "critic_loss": disc_loss,
            "generator_loss": gen_loss_gan,
            "feature_matching_loss": fm_loss,
            "critic_accuracy": critic_accuracy,
            "generator_fool_rate": generator_fool_rate,
            "wasserstein_distance": wasserstein_distance,
            "gradient_penalty": gp,
            "reconstruction_loss": reconstruction_loss,
            "generator_grad_norm": tf.linalg.global_norm(gen_grads),
            "generator_var_count": tf.cast(len(self.generator.trainable_variables), tf.float32)
        }

    # -------------------------------------------------------
    # Helpers for external metric access or resetting (optional)
    # -------------------------------------------------------
    @property
    def metrics(self):
        return [
            self.metric_disc, 
            self.metric_gen, 
            self.metric_fm,
            self.metric_critic_accuracy,
            self.metric_generator_fool_rate,
            self.metric_wasserstein,
            self.metric_gp,
            self.metric_recon
        ]
        
    def _initialize_optimizer_slots(self):
        """Initialize optimizer slots to avoid errors when loading checkpoints"""
        # Create dummy variables and gradients
        dummy_vars_gen = [tf.zeros_like(var) for var in self.generator.trainable_variables]
        dummy_vars_disc = [tf.zeros_like(var) for var in self.discriminator.trainable_variables]
        dummy_vars_ae = [tf.zeros_like(var) for var in self.autoencoder.trainable_variables]
        
        # Apply dummy gradients to create optimizer slots
        self.gen_optimizer.apply_gradients(zip(dummy_vars_gen, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(dummy_vars_disc, self.discriminator.trainable_variables))
        self.ae_optimizer.apply_gradients(zip(dummy_vars_ae, self.autoencoder.trainable_variables))
