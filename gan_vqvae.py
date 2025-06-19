import tensorflow as tf
import tensorflow.keras.backend as K
import spektral
from spektral.utils import normalized_laplacian, rescale_laplacian

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
from pulse import *
from generator import *

from autoencoder import VQVariationalAutoencoder
from autoencoder_analysis import plot_3d_scatter_with_profiles

class GraphVQVariationalAutoencoderGAN(VQVariationalAutoencoder):
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
        generator_layer_sizes=[],
        critic_layer_sizes=[],
        name='gvqvae-gan'
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
        self.generator_layer_sizes = generator_layer_sizes
        self.critic_layer_sizes = critic_layer_sizes
        
        dropout_rate = 0.2
        use_batchnorm = True
        
        # Create DeepGCN encoder with residual connections and dilated convolutions
        self.encoder_layers = []
        self.encoder_blocks = []
        
        # Precompute dilated adjacency matrices
        self.dilated_adj_matrices = {}
        dilation_rates = [1, 2, 4, 8]
        for rate in dilation_rates:
            if rate == 1:
                self.dilated_adj_matrices[rate] = self.adjacency_matrix
            else:
                # Approximate dilated convolution through powers of the adjacency matrix
                # A^k connects nodes that are k steps apart
                dilated_adj = self.adjacency_matrix
                for _ in range(rate - 1):
                    dilated_adj = tf.matmul(dilated_adj, self.adjacency_matrix)
                # Normalize the dilated adjacency matrix
                dilated_adj = normalized_laplacian(dilated_adj.numpy())
                dilated_adj = rescale_laplacian(dilated_adj, lmax=2)
                self.dilated_adj_matrices[rate] = dilated_adj
        
        # Initial convolution
        if encoder_layer_sizes:
            self.encoder_layers.append(spektral.layers.GCNConv(encoder_layer_sizes[0], activation='relu'))
            
            # Add DeepGCN blocks
            for i, units in enumerate(encoder_layer_sizes[1:], 1):
                # Create a residual block
                block_layers = []
                
                # Dilated convolution
                dilation_rate = dilation_rates[i % len(dilation_rates)]
                block_layers.append((spektral.layers.GCNConv(units, activation='relu'), dilation_rate))
                
                if dropout_rate > 0:
                    block_layers.append(tf.keras.layers.Dropout(dropout_rate))
                if use_batchnorm:
                    block_layers.append(tf.keras.layers.BatchNormalization())
                    
                self.encoder_blocks.append(block_layers)
        
        # Output layer for VAE (mean and log_var)
        self.encoder_layers.append(spektral.layers.GCNConv(latent_dim * 2))
        
        # Create DeepGCN decoder with residual connections and dilated convolutions
        self.decoder_layers = []
        self.decoder_blocks = []
        
        # Initial convolution
        if decoder_layer_sizes:
            self.decoder_layers.append(spektral.layers.GCNConv(decoder_layer_sizes[0], activation='relu'))
            
            # Add DeepGCN blocks
            for i, units in enumerate(decoder_layer_sizes[1:], 1):
                # Create a residual block
                block_layers = []
                
                # Dilated convolution
                dilation_rate = dilation_rates[i % len(dilation_rates)]
                block_layers.append((spektral.layers.GCNConv(units, activation='relu'), dilation_rate))
                
                if dropout_rate > 0:
                    block_layers.append(tf.keras.layers.Dropout(dropout_rate))
                if use_batchnorm:
                    block_layers.append(tf.keras.layers.BatchNormalization())
                    
                self.decoder_blocks.append(block_layers)
        
        # Output layer for reconstruction
        self.decoder_layers.append(spektral.layers.GCNConv(input_shape[1], activation='softplus'))

        # Create Generator (from noise to graph data)
        self.generator_layers = []
        self.generator_blocks = []
        
        # Initial generator layer
        if generator_layer_sizes:
            self.generator_layers.append(spektral.layers.GCNConv(generator_layer_sizes[0], activation='relu'))
            
            # Add generator blocks
            for i, units in enumerate(generator_layer_sizes[1:], 1):
                # Create a residual block
                block_layers = []
                
                # Dilated convolution
                dilation_rate = dilation_rates[i % len(dilation_rates)]
                block_layers.append((spektral.layers.GCNConv(units, activation='relu'), dilation_rate))
                
                if dropout_rate > 0:
                    block_layers.append(tf.keras.layers.Dropout(dropout_rate))
                if use_batchnorm:
                    block_layers.append(tf.keras.layers.BatchNormalization())
                    
                self.generator_blocks.append(block_layers)
        
        # Output layer for generator
        self.generator_layers.append(spektral.layers.GCNConv(input_shape[1], activation='softplus'))

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
        self.generator_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.discriminator_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")
        
    def encode(self, x):
        # Initial layers
        for layer in self.encoder_layers[:-1]:
            x = layer([x, self.adjacency_matrix])
            
        # Process through residual blocks
        for block in self.encoder_blocks:
            residual = x
            for layer in block:
                if isinstance(layer, tuple) and isinstance(layer[0], spektral.layers.GCNConv):
                    gcn_layer, dilation_rate = layer
                    # Use the appropriate dilated adjacency matrix
                    x = gcn_layer([x, self.dilated_adj_matrices[dilation_rate]])
                elif isinstance(layer, spektral.layers.GCNConv):
                    x = layer([x, self.adjacency_matrix])
                else:
                    x = layer(x)
            # Add residual connection
            x = x + residual
        
        # Final layer to get mean and log_var
        x = self.encoder_layers[-1]([x, self.adjacency_matrix])
        mean, log_var = tf.split(x, num_or_size_splits=2, axis=-1)
        return mean, log_var
    
    def decode(self, z):
        # Initial layers
        if len(self.decoder_layers) > 1:
            z = self.decoder_layers[0]([z, self.adjacency_matrix])
            
        # Process through residual blocks
        for block in self.decoder_blocks:
            residual = z
            for layer in block:
                if isinstance(layer, tuple) and isinstance(layer[0], spektral.layers.GCNConv):
                    gcn_layer, dilation_rate = layer
                    # Use the appropriate dilated adjacency matrix
                    z = gcn_layer([z, self.dilated_adj_matrices[dilation_rate]])
                elif isinstance(layer, spektral.layers.GCNConv):
                    z = layer([z, self.adjacency_matrix])
                else:
                    z = layer(z)
            # Add residual connection
            z = z + residual
        
        # Final layer
        z = self.decoder_layers[-1]([z, self.adjacency_matrix])
        return z
    
    def generate(self, z):
        # Initial generator layers
        if self.generator_layers:
            z = self.generator_layers[0]([z, self.adjacency_matrix])
            
        # Process through generator blocks
        for block in self.generator_blocks:
            residual = z
            for layer in block:
                if isinstance(layer, tuple) and isinstance(layer[0], spektral.layers.GCNConv):
                    gcn_layer, dilation_rate = layer
                    # Use the appropriate dilated adjacency matrix
                    z = gcn_layer([z, self.dilated_adj_matrices[dilation_rate]])
                elif isinstance(layer, spektral.layers.GCNConv):
                    z = layer([z, self.adjacency_matrix])
                else:
                    z = layer(z)
            # Add residual connection
            z = z + residual
        
        # Final generator layer
        z = self.generator_layers[-1]([z, self.adjacency_matrix])
        return z
        
    def sample(self, batch_size=1):
        # Sample from standard normal distribution
        z = tf.random.normal(shape=(batch_size, self.input_shape[0], self.latent_dim))
        # Generate data using the generator
        return self.generate(z)
    
    def summary(self):
        print(f'\n\033[1mModel: {self.name}\033[0m')
        print("\nEncoder:")
        print(f'{"Layer":<30} {"Output Shape":<20} {"Params":<10} {"Dilation":<10}')
        print('-' * 70)
        
        # Print initial encoder layers
        for i, layer in enumerate(self.encoder_layers[:-1]):
            if isinstance(layer, spektral.layers.GCNConv):
                print(f'{layer.name:<30} {"?":<20} {layer.count_params():<10} {1:<10}')
            else:
                print(f'{layer.name:<30} {"?":<20} {layer.count_params():<10} {"-":<10}')
        
        # Print encoder blocks (residual connections)
        for i, block in enumerate(self.encoder_blocks):
            print(f'Residual Block {i+1}:')
            for layer in block:
                if isinstance(layer, tuple) and isinstance(layer[0], spektral.layers.GCNConv):
                    gcn_layer, dilation_rate = layer
                    print(f'{gcn_layer.name:<30} {"?":<20} {gcn_layer.count_params():<10} {dilation_rate:<10}')
                elif isinstance(layer, spektral.layers.GCNConv):
                    print(f'{layer.name:<30} {"?":<20} {layer.count_params():<10} {1:<10}')
                else:
                    print(f'{layer.name:<30} {"?":<20} {layer.count_params():<10} {"-":<10}')
        
        # Print final encoder layer
        final_layer = self.encoder_layers[-1]
        print(f'{final_layer.name:<30} {"?":<20} {final_layer.count_params():<10} {1:<10}')
        
        print("\nDecoder:")
        print(f'{"Layer":<30} {"Output Shape":<20} {"Params":<10} {"Dilation":<10}')
        print('-' * 70)
        
        # Print initial decoder layers
        if len(self.decoder_layers) > 1:
            layer = self.decoder_layers[0]
            print(f'{layer.name:<30} {"?":<20} {layer.count_params():<10} {1:<10}')
        
        # Print decoder blocks (residual connections)
        for i, block in enumerate(self.decoder_blocks):
            print(f'Residual Block {i+1}:')
            for layer in block:
                if isinstance(layer, tuple) and isinstance(layer[0], spektral.layers.GCNConv):
                    gcn_layer, dilation_rate = layer
                    print(f'{gcn_layer.name:<30} {"?":<20} {gcn_layer.count_params():<10} {dilation_rate:<10}')
                elif isinstance(layer, spektral.layers.GCNConv):
                    print(f'{layer.name:<30} {"?":<20} {layer.count_params():<10} {1:<10}')
                else:
                    print(f'{layer.name:<30} {"?":<20} {layer.count_params():<10} {"-":<10}')
        
        # Print final decoder layer
        final_layer = self.decoder_layers[-1]
        print(f'{final_layer.name:<30} {"?":<20} {final_layer.count_params():<10} {1:<10}')
        
        print("\nGenerator:")
        print(f'{"Layer":<30} {"Output Shape":<20} {"Params":<10} {"Dilation":<10}')
        print('-' * 70)
        
        # Print initial generator layers
        if self.generator_layers:
            layer = self.generator_layers[0]
            print(f'{layer.name:<30} {"?":<20} {layer.count_params():<10} {1:<10}')
        
        # Print generator blocks (residual connections)
        for i, block in enumerate(self.generator_blocks):
            print(f'Residual Block {i+1}:')
            for layer in block:
                if isinstance(layer, tuple) and isinstance(layer[0], spektral.layers.GCNConv):
                    gcn_layer, dilation_rate = layer
                    print(f'{gcn_layer.name:<30} {"?":<20} {gcn_layer.count_params():<10} {dilation_rate:<10}')
                elif isinstance(layer, spektral.layers.GCNConv):
                    print(f'{layer.name:<30} {"?":<20} {layer.count_params():<10} {1:<10}')
                else:
                    print(f'{layer.name:<30} {"?":<20} {layer.count_params():<10} {"-":<10}')
        
        # Print final generator layer
        if self.generator_layers:
            final_layer = self.generator_layers[-1]
            print(f'{final_layer.name:<30} {"?":<20} {final_layer.count_params():<10} {1:<10}')
        
        print()


class GraphCritic(tf.keras.Model):
    def __init__(
        self,
        input_shape,
        adjacency_matrix,
        layer_sizes=[],
        dropout_rate=0.2,
        use_batchnorm=True,
        name='graph_critic'
    ):
        super().__init__(name=name)
        
        adjacency_matrix = normalized_laplacian(adjacency_matrix)
        adjacency_matrix = rescale_laplacian(adjacency_matrix, lmax=2)
        
        self.adjacency_matrix = adjacency_matrix
        self.input_shape = input_shape
        self.layer_sizes = layer_sizes
        
        # Create DeepGCN critic with residual connections and dilated convolutions
        self.layers_list = []
        self.blocks = []
        
        # Precompute dilated adjacency matrices
        self.dilated_adj_matrices = {}
        dilation_rates = [1, 2, 4, 8]
        for rate in dilation_rates:
            if rate == 1:
                self.dilated_adj_matrices[rate] = self.adjacency_matrix
            else:
                # Approximate dilated convolution through powers of the adjacency matrix
                # A^k connects nodes that are k steps apart
                dilated_adj = self.adjacency_matrix
                for _ in range(rate - 1):
                    dilated_adj = tf.matmul(dilated_adj, self.adjacency_matrix)
                # Normalize the dilated adjacency matrix
                dilated_adj = normalized_laplacian(dilated_adj.numpy())
                dilated_adj = rescale_laplacian(dilated_adj, lmax=2)
                self.dilated_adj_matrices[rate] = dilated_adj
        
        # Initial convolution
        if layer_sizes:
            self.layers_list.append(spektral.layers.GCNConv(layer_sizes[0], activation='relu'))
            
            # Add DeepGCN blocks
            for i, units in enumerate(layer_sizes[1:], 1):
                # Create a residual block
                block_layers = []
                
                # Dilated convolution
                dilation_rate = dilation_rates[i % len(dilation_rates)]
                block_layers.append((spektral.layers.GCNConv(units, activation='relu'), dilation_rate))
                
                if dropout_rate > 0:
                    block_layers.append(tf.keras.layers.Dropout(dropout_rate))
                if use_batchnorm:
                    block_layers.append(tf.keras.layers.BatchNormalization())
                    
                self.blocks.append(block_layers)
        
        # Output layer (single neuron for binary classification)
        self.layers_list.append(spektral.layers.GCNConv(1))
        # Add global pooling to get a single value per graph
        self.global_pool = spektral.layers.GlobalSumPool()
        
        # Fully connected layers for conditioning
        self.condition_layer1 = tf.keras.layers.Dense(64, activation='relu')
        self.condition_layer2 = tf.keras.layers.Dense(1)
        
    def call(self, inputs, training=False):
        x, condition = inputs
        
        # Process the main input through GCN layers
        for layer in self.layers_list[:-1]:
            x = layer([x, self.adjacency_matrix])
            
        # Process through residual blocks
        for block in self.blocks:
            residual = x
            for layer in block:
                if isinstance(layer, tuple) and isinstance(layer[0], spektral.layers.GCNConv):
                    gcn_layer, dilation_rate = layer
                    # Use the appropriate dilated adjacency matrix
                    x = gcn_layer([x, self.dilated_adj_matrices[dilation_rate]])
                elif isinstance(layer, spektral.layers.GCNConv):
                    x = layer([x, self.adjacency_matrix])
                else:
                    x = layer(x)
            # Add residual connection
            x = x + residual
        
        # Final layer
        x = self.layers_list[-1]([x, self.adjacency_matrix])
        
        # Global pooling to get a graph-level representation
        x = self.global_pool(x)  # Shape: [batch_size, 1]
        
        # Process the condition through dense layers
        # First pool the condition to get a graph-level representation
        condition_pooled = self.global_pool(condition)  # Shape: [batch_size, 1]
        condition = self.condition_layer1(condition_pooled)  # Shape: [batch_size, 64]
        condition = self.condition_layer2(condition)  # Shape: [batch_size, 1]
        
        # Concatenate the main input processing with the condition
        # This allows the critic to learn how to combine both pieces of information
        output = tf.concat([x, condition], axis=-1)  # Shape: [batch_size, 2]
        
        # Final dense layer to produce a single score
        output = tf.keras.layers.Dense(1)(output)  # Shape: [batch_size, 1]
        
        return output
        
    def summary(self):
        print(f'\n\033[1mModel: {self.name}\033[0m')
        print(f'{"Layer":<30} {"Output Shape":<20} {"Params":<10} {"Dilation":<10}')
        print('-' * 70)
        
        # Print initial layers
        for i, layer in enumerate(self.layers_list[:-1]):
            if isinstance(layer, spektral.layers.GCNConv):
                print(f'{layer.name:<30} {"?":<20} {layer.count_params():<10} {1:<10}')
            else:
                print(f'{layer.name:<30} {"?":<20} {layer.count_params():<10} {"-":<10}')
        
        # Print blocks (residual connections)
        for i, block in enumerate(self.blocks):
            print(f'Residual Block {i+1}:')
            for layer in block:
                if isinstance(layer, tuple) and isinstance(layer[0], spektral.layers.GCNConv):
                    gcn_layer, dilation_rate = layer
                    print(f'{gcn_layer.name:<30} {"?":<20} {gcn_layer.count_params():<10} {dilation_rate:<10}')
                elif isinstance(layer, spektral.layers.GCNConv):
                    print(f'{layer.name:<30} {"?":<20} {layer.count_params():<10} {1:<10}')
                else:
                    print(f'{layer.name:<30} {"?":<20} {layer.count_params():<10} {"-":<10}')
        
        # Print final layer
        final_layer = self.layers_list[-1]
        print(f'{final_layer.name:<30} {"?":<20} {final_layer.count_params():<10} {1:<10}')
        
        # Print global pooling and conditional layers
        print(f'{self.global_pool.name:<30} {"?":<20} {self.global_pool.count_params():<10} {"-":<10}')
        print(f'{self.condition_layer1.name:<30} {"?":<20} {self.condition_layer1.count_params():<10} {"-":<10}')
        print(f'{self.condition_layer2.name:<30} {"?":<20} {self.condition_layer2.count_params():<10} {"-":<10}')
        
        print()


class GraphVQVAEGANTrainer:
    def __init__(
        self,
        vae_gan,
        critic,
        adjacency_matrix,
        reconstruction_weight=1.0,
        commitment_weight=0.25,
        generator_weight=1.0,
        critic_weight=1.0,
        critic_steps=5
    ):
        self.vae_gan = vae_gan
        self.critic = critic
        self.adjacency_matrix = adjacency_matrix
        self.reconstruction_weight = reconstruction_weight
        self.commitment_weight = commitment_weight
        self.generator_weight = generator_weight
        self.critic_weight = critic_weight
        self.critic_steps = critic_steps
        
        # Optimizers
        self.vae_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        
    @tf.function
    def train_step(self, real_data, condition_data):
        batch_size = tf.shape(real_data)[0]
        
        # Train critic multiple steps
        for _ in range(self.critic_steps):
            # Generate fake data
            noise = tf.random.normal(shape=(batch_size, self.vae_gan.input_shape[0], self.vae_gan.latent_dim))
            fake_data = self.vae_gan.generate(noise)
            
            with tf.GradientTape() as critic_tape:
                # Critic loss on real data
                real_output = self.critic([real_data, condition_data], training=True)
                # Critic loss on fake data
                fake_output = self.critic([fake_data, condition_data], training=True)
                
                # Wasserstein loss
                critic_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
                
                # Gradient penalty (improved WGAN)
                alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
                interpolated = alpha * real_data + (1 - alpha) * fake_data
                
                with tf.GradientTape() as gp_tape:
                    gp_tape.watch(interpolated)
                    # Get critic output for interpolated data
                    interpolated_output = self.critic([interpolated, condition_data], training=True)
                
                # Calculate gradients with respect to interpolated data
                grads = gp_tape.gradient(interpolated_output, interpolated)
                # Calculate the norm of the gradients
                grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
                gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1.0))
                
                # Apply gradient penalty to critic loss
                critic_loss = critic_loss + 10.0 * gradient_penalty
            
            # Apply gradients to critic
            critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        # Train VAE and generator
        with tf.GradientTape() as vae_tape, tf.GradientTape() as gen_tape:
            # VAE forward pass
            mean, log_var = self.vae_gan.encode(real_data)
            z = self.vae_gan.reparameterize(mean, log_var)
            
            # Quantize
            z_e_expand = tf.expand_dims(z, 2)
            codebook_expand = tf.reshape(
                self.vae_gan.codebook, [1, 1, self.vae_gan.num_embeddings, self.vae_gan.embedding_dim]
            )
            distances = tf.reduce_sum((z_e_expand - codebook_expand) ** 2, axis=-1)
            encoding_indices = tf.argmin(distances, axis=-1)
            z_q = tf.gather(self.vae_gan.codebook, encoding_indices, axis=0)
            z_q_st = z + tf.stop_gradient(z_q - z)
            
            # Decode for reconstruction
            reconstructed = self.vae_gan.decode(z_q_st)
            
            # Generate fake data for generator training
            noise = tf.random.normal(shape=(batch_size, self.vae_gan.input_shape[0], self.vae_gan.latent_dim))
            fake_data = self.vae_gan.generate(noise)
            
            # Calculate losses
            # Reconstruction loss (MSE)
            reconstruction_loss = tf.reduce_mean(tf.square(real_data - reconstructed))
            
            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
            
            # Commitment loss
            vq_loss = tf.reduce_mean(tf.square(tf.stop_gradient(z) - z_q)) + tf.reduce_mean(tf.square(z - tf.stop_gradient(z_q)))
            
            # Generator loss (fool the critic)
            fake_output = self.critic([fake_data, condition_data], training=False)
            generator_loss = -tf.reduce_mean(fake_output)
            
            # Total VAE loss
            vae_loss = self.reconstruction_weight * reconstruction_loss + kl_loss + self.commitment_weight * vq_loss
            
            # Generator loss (separate from VAE)
            gen_loss = self.generator_weight * generator_loss
        
        # Apply gradients to VAE
        vae_grads = vae_tape.gradient(vae_loss, self.vae_gan.trainable_variables)
        self.vae_optimizer.apply_gradients(zip(vae_grads, self.vae_gan.trainable_variables))
        
        # Apply gradients to generator (only to generator layers)
        generator_vars = [var for var in self.vae_gan.trainable_variables if 'generator' in var.name]
        gen_grads = gen_tape.gradient(gen_loss, generator_vars)
        self.generator_optimizer.apply_gradients(zip(gen_grads, generator_vars))
        
        return {
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss,
            'vq_loss': vq_loss,
            'generator_loss': generator_loss,
            'critic_loss': critic_loss
        }