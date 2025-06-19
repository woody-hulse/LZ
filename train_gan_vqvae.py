import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import spektral
from spektral.utils import normalized_laplacian, rescale_laplacian

from gan import GraphVQVariationalAutoencoderGAN, GraphCritic, GraphVQVAEGANTrainer
from simple_pulse import vertex_electron_batch_generator, Params
from simple_pulse import generate_vertex_electron_params, simulate_fixed_vertex_electron
from autoencoder_analysis import plot_3d_scatter_with_profiles
from preprocessing import create_grid_adjacency
import networkx as nx
import time


def train_graph_vqvae_gan(
    # Model parameters
    input_shape=(24, 24, 700),  # (R, C, T)
    latent_dim=16,
    num_embeddings=64,
    embedding_dim=16,
    commitment_cost=0.25,
    encoder_layer_sizes=[32, 64, 128],
    decoder_layer_sizes=[128, 64, 32],
    generator_layer_sizes=[32, 64, 32],
    critic_layer_sizes=[32, 64, 32],
    
    # Training parameters
    batch_size=32,
    vae_epochs=100,  # Train VAE to convergence
    gan_epochs=100,  # Then train GAN
    reconstruction_weight=1.0,
    commitment_weight=0.25,
    generator_weight=1.0,
    critic_weight=1.0,
    critic_steps=5,
    
    # Data parameters
    n_list=[1, 2, 3, 4],
    
    # Directories
    log_dir='logs',
    checkpoint_dir='ckpts',
    sample_dir='samples'
):
    """
    Train a Graph VQVAE GAN model using vertex electron data
    
    Parameters:
    -----------
    input_shape: tuple
        Shape of the input data (R, C, T)
    latent_dim: int
        Dimension of the latent space
    num_embeddings: int
        Number of embeddings in the codebook
    embedding_dim: int
        Dimension of each embedding in the codebook
    commitment_cost: float
        Weight for the commitment loss
    encoder_layer_sizes: list
        List of layer sizes for the encoder
    decoder_layer_sizes: list
        List of layer sizes for the decoder
    generator_layer_sizes: list
        List of layer sizes for the generator
    critic_layer_sizes: list
        List of layer sizes for the critic
    batch_size: int
        Batch size for training
    vae_epochs: int
        Number of epochs for VAE training
    gan_epochs: int
        Number of epochs for GAN training
    n_list: list
        List of possible numbers of vertices
    """
    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # Setup tensorboard logging
    current_time = time.strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(log_dir, current_time)
    summary_writer = tf.summary.create_file_writer(train_log_dir)
    
    # Setup simulation parameters
    params = Params()
    assert input_shape == (params.R, params.C, params.T)

    # Create the graph adjacency matrix for spatial dimensions (R x C)
    # The adjacency matrix has shape (R*C, R*C)
    adjacency_matrix = create_grid_adjacency(input_shape[1])
    
    # Create the model
    vae_gan = GraphVQVariationalAutoencoderGAN(
        input_shape=(input_shape[0] * input_shape[1], input_shape[2]),  # Shape (R*C, T)
        latent_dim=latent_dim,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=commitment_cost,
        adjacency_matrix=adjacency_matrix,
        encoder_layer_sizes=encoder_layer_sizes,
        decoder_layer_sizes=decoder_layer_sizes,
        generator_layer_sizes=generator_layer_sizes,
        critic_layer_sizes=critic_layer_sizes
    )
    
    # Create the critic
    critic = GraphCritic(
        input_shape=(input_shape[0] * input_shape[1], input_shape[2]),  # Shape (R*C, T)
        adjacency_matrix=adjacency_matrix,
        layer_sizes=critic_layer_sizes
    )
    
    # Create the trainer
    trainer = GraphVQVAEGANTrainer(
        vae_gan=vae_gan,
        critic=critic,
        adjacency_matrix=adjacency_matrix,
        reconstruction_weight=reconstruction_weight,
        commitment_weight=commitment_weight,
        generator_weight=generator_weight,
        critic_weight=critic_weight,
        critic_steps=critic_steps
    )
    
    # Setup checkpoints
    vae_gan_ckpt = tf.train.Checkpoint(model=vae_gan, optimizer=trainer.vae_optimizer)
    critic_ckpt = tf.train.Checkpoint(model=critic, optimizer=trainer.critic_optimizer)
    
    vae_gan_manager = tf.train.CheckpointManager(
        vae_gan_ckpt, 
        directory=os.path.join(checkpoint_dir, 'vae_gan'), 
        max_to_keep=3
    )
    critic_manager = tf.train.CheckpointManager(
        critic_ckpt, 
        directory=os.path.join(checkpoint_dir, 'critic'), 
        max_to_keep=3
    )
    
    # Restore from latest checkpoint if available
    if vae_gan_manager.latest_checkpoint:
        vae_gan_ckpt.restore(vae_gan_manager.latest_checkpoint).assert_existing_objects_matched()
        print('Restored VAE-GAN from checkpoint:', vae_gan_manager.latest_checkpoint)
    if critic_manager.latest_checkpoint:
        critic_ckpt.restore(critic_manager.latest_checkpoint).assert_existing_objects_matched()
        print('Restored critic from checkpoint:', critic_manager.latest_checkpoint)
    
    # Phase 1: Train VAE to convergence
    print("\nPhase 1: Training VAE to convergence...")
    
    # Create data generators
    def train_data_generator():
        while True:
            batch1_pulses, _ = generate_batches(batch_size, input_shape, n_list, params)
            batch1_reshaped = np.reshape(batch1_pulses, (batch_size, input_shape[0] * input_shape[1], input_shape[2]))
            yield batch1_reshaped, batch1_reshaped  # Autoencoder input and target are the same

    def val_data_generator():
        while True:
            batch1_pulses, _ = generate_batches(batch_size, input_shape, n_list, params)
            batch1_reshaped = np.reshape(batch1_pulses, (batch_size, input_shape[0] * input_shape[1], input_shape[2]))
            yield batch1_reshaped, batch1_reshaped

    # Compile VAE model
    vae_gan.compile(
        optimizer=trainer.vae_optimizer,
        loss=lambda y_true, y_pred: reconstruction_weight * tf.reduce_mean(tf.square(y_true - y_pred))
    )
    
    # Build model
    batch_x, _ = next(train_data_generator())
    vae_gan.build(batch_x.shape[1:])
    # vae_gan.summary()

    # Setup checkpointing
    vae_ckpt = tf.train.Checkpoint(model=vae_gan, optimizer=trainer.vae_optimizer)
    vae_ckpt_manager = tf.train.CheckpointManager(
        vae_ckpt, 
        directory=os.path.join(checkpoint_dir, 'vae'), 
        max_to_keep=3
    )

    # Restore from checkpoint if available
    if vae_ckpt_manager.latest_checkpoint:
        vae_ckpt.restore(vae_ckpt_manager.latest_checkpoint).assert_existing_objects_matched()
        print('Restored VAE from checkpoint:', vae_ckpt_manager.latest_checkpoint)

    # Custom callback for checkpointing and early stopping
    class VAECallback(tf.keras.callbacks.Callback):
        def __init__(self, ckpt_manager):
            super().__init__()
            self.ckpt_manager = ckpt_manager
            self.best_loss = float('inf')
            self.wait = 0
            self.patience = 10

        def on_epoch_end(self, epoch, logs=None):
            current_loss = logs.get('val_loss')
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.wait = 0
                self.ckpt_manager.save()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.model.stop_training = True
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")

    # Train VAE
    history = vae_gan.fit(
        train_data_generator(),
        epochs=vae_epochs,
        steps_per_epoch=100,  # Number of batches per epoch
        validation_data=val_data_generator(),
        validation_steps=4,  # Number of validation batches
        callbacks=[VAECallback(vae_ckpt_manager)]
    )

    # Freeze encoder weights
    print("\nFreezing encoder weights...")
    for layer in vae_gan.encoder_layers:
        layer.trainable = False
    for block in vae_gan.encoder_blocks:
        for layer in block:
            if isinstance(layer, tuple):
                layer[0].trainable = False
            else:
                layer.trainable = False
    
    # Phase 2: Train GAN with frozen encoder
    print("\nPhase 2: Training GAN with frozen encoder...")
    for epoch in range(gan_epochs):
        start_time = time.time()
        
        # Generate batches
        batch1_pulses, batch2_pulses = generate_batches(batch_size, input_shape, n_list, params)
        batch1_reshaped = np.reshape(batch1_pulses, (batch_size, input_shape[0] * input_shape[1], input_shape[2]))
        batch2_reshaped = np.reshape(batch2_pulses, (batch_size, input_shape[0] * input_shape[1], input_shape[2]))
        
        # Train GAN
        metrics = custom_train_step(trainer, batch1_reshaped, batch2_reshaped)
        
        # Log metrics
        with summary_writer.as_default():
            tf.summary.scalar('gan_generator_loss', metrics['generator_loss'], step=epoch + vae_epochs)
            tf.summary.scalar('gan_critic_loss', metrics['critic_loss'], step=epoch + vae_epochs)
        
        # Print progress
        elapsed_time = time.time() - start_time
        print(f"GAN Epoch {epoch+1}/{gan_epochs}, Time: {elapsed_time:.2f}s")
        print(f"  Generator Loss: {metrics['generator_loss']:.4f}")
        print(f"  Critic Loss: {metrics['critic_loss']:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            vae_gan_manager.save()
            critic_manager.save()
            generate_and_save_samples(
                vae_gan=vae_gan,
                batch_size=5,
                input_shape=input_shape,
                epoch=epoch+1 + vae_epochs,
                sample_dir=sample_dir
            )
    
    return vae_gan, critic


def custom_train_step(trainer, batch1_reshaped, batch2_reshaped):
    """
    Custom training step implementing the procedure:
    1. Batch 1 is fed into the autoencoder-GAN
    2. Batch 1 is used as the conditional for the critic
    3. The critic discriminates between Batch 2 (real) and the autoencoder-GAN output (fake)
    
    Parameters:
    -----------
    trainer: GraphVQVAEGANTrainer
        The trainer instance
    batch1_reshaped: ndarray of shape (batch_size, R*C, T)
        First batch (for autoencoder input and conditional)
    batch2_reshaped: ndarray of shape (batch_size, R*C, T) 
        Second batch (real samples for critic)
    
    Returns:
    --------
    metrics: dict
        Training metrics
    """
    batch_size = len(batch1_reshaped)
    vae_gan = trainer.vae_gan
    critic = trainer.critic
    
    # Train critic multiple steps
    for _ in range(trainer.critic_steps):
        with tf.GradientTape() as critic_tape:
            # Autoencoder forward pass
            mean, log_var = vae_gan.encode(batch1_reshaped)
            z = vae_gan.reparameterize(mean, log_var)
            
            # Quantize
            z_e_expand = tf.expand_dims(z, 2)
            codebook_expand = tf.reshape(
                vae_gan.codebook, [1, 1, vae_gan.num_embeddings, vae_gan.embedding_dim]
            )
            distances = tf.reduce_sum((z_e_expand - codebook_expand) ** 2, axis=-1)
            encoding_indices = tf.argmin(distances, axis=-1)
            z_q = tf.gather(vae_gan.codebook, encoding_indices, axis=0)
            z_q_st = z + tf.stop_gradient(z_q - z)
            
            # Decode for reconstruction - this is the "fake" data
            fake_data = vae_gan.decode(z_q_st)
            
            # Critic loss on real data (batch2_reshaped) conditioned on batch1_reshaped
            real_output = critic([batch2_reshaped, batch1_reshaped], training=True)
            
            # Critic loss on fake data (reconstructed) conditioned on batch1_reshaped
            fake_output = critic([fake_data, batch1_reshaped], training=True)
            
            # Wasserstein loss
            critic_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
            
            # Gradient penalty (improved WGAN)
            alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
            interpolated = alpha * batch2_reshaped + (1 - alpha) * fake_data
            
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                # Get critic output for interpolated data
                interpolated_output = critic([interpolated, batch1_reshaped], training=True)
            
            # Calculate gradients with respect to interpolated data
            grads = gp_tape.gradient(interpolated_output, interpolated)
            # Calculate the norm of the gradients
            grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
            gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1.0))
            
            # Apply gradient penalty to critic loss
            critic_loss = critic_loss + 10.0 * gradient_penalty
        
        # Apply gradients to critic
        critic_grads = critic_tape.gradient(critic_loss, critic.trainable_variables)
        trainer.critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
    
    # Train VAE and generator
    with tf.GradientTape() as vae_tape, tf.GradientTape() as gen_tape:
        # VAE forward pass
        mean, log_var = vae_gan.encode(batch1_reshaped)
        z = vae_gan.reparameterize(mean, log_var)
        
        # Quantize
        z_e_expand = tf.expand_dims(z, 2)
        codebook_expand = tf.reshape(
            vae_gan.codebook, [1, 1, vae_gan.num_embeddings, vae_gan.embedding_dim]
        )
        distances = tf.reduce_sum((z_e_expand - codebook_expand) ** 2, axis=-1)
        encoding_indices = tf.argmin(distances, axis=-1)
        z_q = tf.gather(vae_gan.codebook, encoding_indices, axis=0)
        z_q_st = z + tf.stop_gradient(z_q - z)
        
        # Decode for reconstruction - this is both the reconstructed data and the "fake" data
        reconstructed = vae_gan.decode(z_q_st)
        
        # Generate samples from noise for the generator (separate from reconstruction)
        noise = tf.random.normal(shape=(batch_size, vae_gan.input_shape[0], vae_gan.latent_dim))
        generated_fake = vae_gan.generate(noise)
        
        # Calculate losses
        # Reconstruction loss (MSE) for the autoencoder
        reconstruction_loss = tf.reduce_mean(tf.square(batch1_reshaped - reconstructed))
        
        # KL divergence
        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
        
        # Commitment loss
        vq_loss = tf.reduce_mean(tf.square(tf.stop_gradient(z) - z_q)) + tf.reduce_mean(tf.square(z - tf.stop_gradient(z_q)))
        
        # Generator loss for the generated samples (fool the critic)
        # The critic is conditioned on batch1_reshaped in both cases
        reconstructed_output = critic([reconstructed, batch1_reshaped], training=False)
        generated_output = critic([generated_fake, batch1_reshaped], training=False)
        
        # We want to fool the critic with both reconstructed and generated samples
        reconstruction_generator_loss = -tf.reduce_mean(reconstructed_output)
        pure_generator_loss = -tf.reduce_mean(generated_output)
        generator_loss = reconstruction_generator_loss + pure_generator_loss
        
        # Total VAE loss (includes reconstruction quality and latent space regularization)
        vae_loss = trainer.reconstruction_weight * reconstruction_loss + kl_loss + trainer.commitment_weight * vq_loss
        
        # Generator loss (separate from VAE, focused on fooling the critic)
        gen_loss = trainer.generator_weight * generator_loss
    
    # Apply gradients to VAE
    vae_grads = vae_tape.gradient(vae_loss, vae_gan.trainable_variables)
    trainer.vae_optimizer.apply_gradients(zip(vae_grads, vae_gan.trainable_variables))
    
    # Apply gradients to generator (only to generator layers)
    generator_vars = []
    for var in vae_gan.trainable_variables:
        if any(name in var.name for name in ['generator', 'generate']):
            generator_vars.append(var)
    
    if generator_vars:  # Only apply gradients if we found generator variables
        gen_grads = gen_tape.gradient(gen_loss, generator_vars)
        if gen_grads:  # Only apply gradients if we have gradients
            trainer.generator_optimizer.apply_gradients(zip(gen_grads, generator_vars))
    
    return {
        'generator_loss': generator_loss,
        'critic_loss': critic_loss
    }


def generate_and_save_samples(vae_gan, batch_size, input_shape, epoch, sample_dir):
    """
    Generate samples from the model and save them as images using 3D visualization
    """
    # Generate samples - output will be of shape (batch_size, R*C, T)
    samples = vae_gan.sample(batch_size)
    
    # Reshape samples back to 3D grid (batch_size, R, C, T)
    samples_3d = tf.reshape(samples, (batch_size, input_shape[0], input_shape[1], input_shape[2])).numpy()
    
    # Save each sample as a separate visualization
    for i in range(batch_size):
        # Get the sample
        sample = samples_3d[i]
        
        # Create filename
        sample_path = os.path.join(sample_dir, f"sample_{epoch}_index_{i}.png")
        
        # Plot using 3D scatter with profiles
        fig = plot_3d_scatter_with_profiles(
            event=sample,
            threshold=0.1,  # Use a threshold to filter out low values
            title=f"Generated Sample (Epoch {epoch}, Index {i})",
            figsize=(12, 10),
            colormap='viridis',
            return_fig=True
        )
        
        # Save figure and close
        fig.savefig(sample_path)
        plt.close(fig)
    
    # Also create a composite figure for a quick overview
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
    
    for i in range(batch_size):
        # Get the sample
        sample = samples_3d[i]
        
        # Create projections (sum across each dimension)
        proj_x = np.sum(sample, axis=0)  # Sum across rows (R)
        proj_y = np.sum(sample, axis=1)  # Sum across columns (C)
        proj_z = np.sum(sample, axis=2)  # Sum across time (T)
        
        # Plot projections
        axes[i, 0].imshow(proj_x, cmap='viridis', aspect='auto')
        axes[i, 0].set_title(f"Sample {i+1} - YZ Projection")
        
        axes[i, 1].imshow(proj_y, cmap='viridis', aspect='auto')
        axes[i, 1].set_title(f"Sample {i+1} - XZ Projection")
        
        axes[i, 2].imshow(proj_z, cmap='viridis', aspect='auto')
        axes[i, 2].set_title(f"Sample {i+1} - XY Projection")
    
    plt.tight_layout()
    
    # Save the composite figure
    overview_path = os.path.join(sample_dir, f"samples_overview_epoch_{epoch}.png")
    plt.savefig(overview_path)
    plt.close()


def generate_batches(batch_size, input_shape, n_list, params):
    """Helper function to generate training batches"""
    all_vertex_positions = []
    all_electron_counts = []
    all_N_values = []
    
    for i in range(batch_size):
        vertex_positions, electron_counts, N = generate_vertex_electron_params(
            N_list=n_list, 
            params=params
        )
        all_vertex_positions.append(vertex_positions)
        all_electron_counts.append(electron_counts)
        all_N_values.append(N)
    
    batch1_pulses = []
    batch2_pulses = []
    
    for i in range(batch_size):
        pulse1, _ = simulate_fixed_vertex_electron(
            vertex_positions=all_vertex_positions[i],
            electron_counts=all_electron_counts[i],
            params=params,
            seed=(int(time.time() * 1000) + i) % (2 ** 21)
        )
        
        pulse2, _ = simulate_fixed_vertex_electron(
            vertex_positions=all_vertex_positions[i],
            electron_counts=all_electron_counts[i],
            params=params,
            seed=(int(time.time() * 1000) + i) % (2 ** 22)
        )
        
        batch1_pulses.append(pulse1)
        batch2_pulses.append(pulse2)
    
    return np.array(batch1_pulses), np.array(batch2_pulses)


if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)  # For debugging
    params = Params()
    
    # Train the model
    vae_gan, critic = train_graph_vqvae_gan(
        # Model parameters
        input_shape=(params.R, params.C, params.T),  # (R, C, T)
        latent_dim=700,
        num_embeddings=128,
        embedding_dim=700,
        commitment_cost=0.25,
        encoder_layer_sizes=[700, 700],
        decoder_layer_sizes=[700, 700],
        generator_layer_sizes=[700, 700],
        critic_layer_sizes=[700, 700],
        
        # Training parameters
        batch_size=16,  # Smaller batch size to avoid out-of-memory issues
        vae_epochs=10,
        gan_epochs=10,
        
        # Data parameters
        n_list=[1, 2, 3, 4]
    ) 