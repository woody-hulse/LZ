import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import spektral
from spektral.utils import normalized_laplacian, rescale_laplacian

from gan import *
from simple_pulse import vertex_electron_batch_generator, Params
from simple_pulse import generate_vertex_electron_params, simulate_fixed_vertex_electron
from autoencoder_analysis import plot_3d_scatter_with_profiles, plot_events
from preprocessing import create_grid_adjacency
import networkx as nx
import time


def plot_training_metrics(metrics_history, save_path=None, window_size=20):
    """
    Creates publication-quality plots of WGAN training metrics.
    
    Parameters:
    -----------
    metrics_history : dict
        Dictionary containing lists of metrics recorded during training
    save_path : str, optional
        Path to save the figure
    window_size : int, optional
        Window size for the moving average of loss values
    """
    from matplotlib.ticker import MaxNLocator
    from scipy.ndimage import gaussian_filter1d

    # Create figure with multiple subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14), dpi=150, sharex=True)
    
    # Get epochs
    epochs = np.arange(len(metrics_history['generator_loss']))
    
    # Colors for plotting
    colors = {
        'wasserstein_distance': '#0000ff',  # blue
        'gradient_penalty': '#4c72b0',      # lighter blue
        'generator_loss': '#ffa500',        # orange
        'critic_loss': '#0000ff',           # blue
        'reconstruction_loss': '#2ca02c',   # green
        'critic_accuracy': '#8b0000',       # dark red
        'generator_fool_rate': '#ff4500'    # orange red
    }
    
    # Plot Wasserstein distance and gradient penalty in first subplot
    ax1.plot(epochs, metrics_history['wasserstein_distance'], 
             color=colors['wasserstein_distance'], linewidth=2, 
             marker='', label='Wasserstein Distance')
    
    ax1.plot(epochs, metrics_history['gradient_penalty'], 
             color=colors['gradient_penalty'], linewidth=2, alpha=0.7,
             marker='', label='Gradient Penalty')

    ax1.set_xlim(min(epochs), max(epochs))
    ax1.set_ylabel('Distance / Penalty', fontsize=12)
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('WGAN Training Metrics', fontsize=14)
    
    # Plot loss metrics in second subplot
    for metric in ['generator_loss', 'critic_loss']:#, 'reconstruction_loss']:
        if metric in metrics_history:
            # Convert to numpy array for processing
            values = np.array(metrics_history[metric])
            
            # Apply gaussian smoothing for trend line
            smooth_values = gaussian_filter1d(values, sigma=window_size/3)
            
            # Plot scatter points with transparency
            ax2.scatter(epochs, values, color=colors[metric], alpha=0.2, s=15)
            
            # Plot smooth trend line
            ax2.plot(epochs, smooth_values, color=colors[metric], 
                     linewidth=1.5, label=f'{metric.replace("_", " ").title()}')
    
    ax2.set_xlim(min(epochs), max(epochs))
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, alpha=0.3)
    
    # Set x-axis to show integers
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Plot new metrics in third subplot
    for metric in ['critic_accuracy', 'generator_fool_rate']:
        if metric in metrics_history:
            # Convert to numpy array for processing
            values = np.array(metrics_history[metric])
            
            # Apply gaussian smoothing for trend line
            smooth_values = gaussian_filter1d(values, sigma=window_size/3)
            
            # Plot scatter points with transparency
            ax3.scatter(epochs, values, color=colors[metric], alpha=0.2, s=15)
            
            # Plot smooth trend line
            ax3.plot(epochs, smooth_values, color=colors[metric], 
                     linewidth=1.5, label=f'{metric.replace("_", " ").title()}')
    
    ax3.set_xlim(min(epochs), max(epochs))
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Metric', fontsize=12)
    ax3.legend(loc='upper right', frameon=True)
    ax3.grid(True, alpha=0.3)
    
    # Set x-axis to show integers
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def train_dense_gan(
    # Model parameters
    input_shape=(24, 24, 700),  # (R, C, T)
    latent_dim=16,
    encoder_layer_sizes=[32, 64, 128],
    decoder_layer_sizes=[128, 64, 32],
    generator_layer_sizes=[32, 64, 32],
    critic_layer_sizes=[32, 64, 32],
    
    # Training parameters
    batch_size=32,
    ae_epochs=100,  # Train autoencoder to convergence
    gan_epochs=100,  # Then train GAN
    reconstruction_weight=0.0,
    feature_matching_weight=0.0,
    generator_weight=1.0,
    critic_steps=5,
    
    # Data parameters
    n_list=[1, 2, 3, 4],
    
    # Directories
    log_dir='logs',
    checkpoint_dir='ckpts',
    sample_dir='samples'
):
    """
    Train a Dense Autoencoder GAN model using vertex electron data
    
    Parameters:
    -----------
    input_shape: tuple
        Shape of the input data (R, C, T)
    latent_dim: int
        Dimension of the latent space
    encoder_layer_sizes: list
        List of layer sizes for the encoder
    decoder_layer_sizes: list
        List of layer sizes for the decoder
    generator_layer_sizes: list
        List of layer sizes for the generator
    critic_layer_sizes: list
        List of layer sizes for the discriminator
    batch_size: int
        Batch size for training
    ae_epochs: int
        Number of epochs for autoencoder training
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

    # Create the autoencoder
    autoencoder = Autoencoder(
        input_shape=(input_shape[0] * input_shape[1], input_shape[2]),  # Shape (R*C, T)
        latent_dim=latent_dim,
        encoder_layer_sizes=encoder_layer_sizes,
        decoder_layer_sizes=decoder_layer_sizes,
    )
    
    # Create the generator (separate model)
    # generator = Generator(
    #     output_shape=(input_shape[0] * input_shape[1], input_shape[2]),
    #     latent_dim=latent_dim,
    #     layer_sizes=generator_layer_sizes if generator_layer_sizes else decoder_layer_sizes,
    # )
    generator = ConvDistributionGenerator(
        output_shape=(input_shape[0], input_shape[1], input_shape[2]),
        latent_dim=latent_dim,
        layer_sizes=generator_layer_sizes if generator_layer_sizes else decoder_layer_sizes,
    )
    
    # Create the discriminator
    discriminator = Conv3DDiscriminator(
        input_shape=(input_shape[0], input_shape[1], input_shape[2])
    )
    
    # Create the trainer
    trainer = GANTrainer(
        autoencoder=autoencoder,
        generator=generator,
        discriminator=discriminator,
        reconstruction_weight=reconstruction_weight,
        generator_weight=generator_weight,
        critic_steps=critic_steps,
        feature_matching_weight=feature_matching_weight
    )
    
    # Initialize models with dummy inputs to build them
    dummy_batch = np.random.normal(size=(1, input_shape[0] * input_shape[1], input_shape[2]))
    dummy_condition = np.random.normal(size=(1, max(n_list), 4))
    dummy_batch = dummy_batch.astype(np.float32)
    dummy_condition = dummy_condition.astype(np.float32)
    _ = autoencoder(dummy_batch)  # Initialize autoencoder
    dummy_batch_encoded = autoencoder.encode(dummy_batch)
    _ = generator(dummy_batch_encoded)  # Initialize generator
    _ = discriminator([dummy_batch, dummy_condition])  # Initialize discriminator
    
    # generator.summary()

    # Explicitly initialize optimizer slots to match what's in checkpoints
    trainer._initialize_optimizer_slots()
    
    # Run one training step to fully initialize all variables
    batch1_pulses, batch1_params = generate_batches(batch_size, input_shape, n_list, params)
    batch1_reshaped = np.reshape(batch1_pulses, (batch_size, input_shape[0] * input_shape[1], input_shape[2]))
    trainer.train_step(batch1_reshaped, batch1_params)
    
    # Custom callback for checkpointing
    class CheckpointCallback(tf.keras.callbacks.Callback):
        def __init__(self, ckpt_manager):
            super().__init__()
            self.ckpt_manager = ckpt_manager
        
        def on_epoch_end(self, epoch, logs=None):
            self.ckpt_manager.save()

    # Setup checkpoints
    ae_ckpt = tf.train.Checkpoint(model=autoencoder, optimizer=trainer.ae_optimizer)
    gen_ckpt = tf.train.Checkpoint(model=generator, optimizer=trainer.gen_optimizer)
    critic_ckpt = tf.train.Checkpoint(model=discriminator, optimizer=trainer.disc_optimizer)

    ae_manager = tf.train.CheckpointManager(
        ae_ckpt, 
        directory=os.path.join(checkpoint_dir, autoencoder.name), 
        max_to_keep=3
    )
    gen_manager = tf.train.CheckpointManager(
        gen_ckpt, 
        directory=os.path.join(checkpoint_dir, generator.name), 
        max_to_keep=3
    )
    critic_manager = tf.train.CheckpointManager(
        critic_ckpt, 
        directory=os.path.join(checkpoint_dir, discriminator.name), 
        max_to_keep=3
    )

    # Compile and initialize models
    autoencoder.compile(optimizer=trainer.ae_optimizer, loss=tf.keras.losses.MeanSquaredError())

    # Create data generators
    def train_data_generator():
        while True:
            batch1_pulses, _ = generate_batches(32, input_shape, n_list, params)
            batch1_reshaped = np.reshape(batch1_pulses, (32, input_shape[0] * input_shape[1], input_shape[2]))
            yield batch1_reshaped, batch1_reshaped  # Autoencoder input and target are the same

    def val_data_generator():
        while True:
            batch1_pulses, _ = generate_batches(32, input_shape, n_list, params)
            batch1_reshaped = np.reshape(batch1_pulses, (32, input_shape[0] * input_shape[1], input_shape[2]))
            yield batch1_reshaped, batch1_reshaped

    batch1_pulses, batch1_params = generate_batches(batch_size, input_shape, n_list, params)
    batch1_reshaped = np.reshape(batch1_pulses, (batch_size, input_shape[0] * input_shape[1], input_shape[2]))
    autoencoder.train_on_batch(batch1_reshaped, batch1_reshaped)

    # Restore checkpoints if available
    if ae_manager.latest_checkpoint:
        ae_ckpt.restore(ae_manager.latest_checkpoint).expect_partial()
        print('Restored autoencoder from checkpoint:', ae_manager.latest_checkpoint)
    else:
        print('No autoencoder checkpoint found, initializing from scratch.')

    if gen_manager.latest_checkpoint:
        gen_ckpt.restore(gen_manager.latest_checkpoint).expect_partial()
        print('Restored generator from checkpoint:', gen_manager.latest_checkpoint)
    else:
        print('No generator checkpoint found, initializing from scratch.')

    if critic_manager.latest_checkpoint:
        critic_ckpt.restore(critic_manager.latest_checkpoint).expect_partial()
        print('Restored discriminator from checkpoint:', critic_manager.latest_checkpoint)
    else:
        print('No discriminator checkpoint found, initializing from scratch.')
    
    # Phase 1: Train autoencoder to convergence
    print("\nPhase 1: Training autoencoder to convergence...")
    

    # Build model
    batch_x, _ = next(train_data_generator())
    autoencoder.build(batch_x.shape[1:])
    # autoencoder.summary()

    # Custom callback for checkpointing and early stopping
    class AECallback(tf.keras.callbacks.Callback):
        def __init__(self, ckpt_manager):
            super().__init__()
            self.ckpt_manager = ckpt_manager
            self.best_loss = float('inf')
            self.wait = 0
            self.patience = 10

        def on_epoch_end(self, epoch, logs=None):
            self.ckpt_manager.save()
            current_loss = logs.get('val_loss')
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.model.stop_training = True
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")

    # Train autoencoder
    history = autoencoder.fit(
        train_data_generator(),
        epochs=ae_epochs,
        steps_per_epoch=64,  # Number of batches per epoch
        validation_data=val_data_generator(),
        validation_steps=4,  # Number of validation batches
        callbacks=[AECallback(ae_manager)]
    )

    # Freeze encoder weights
    print("\nFreezing encoder weights...")
    for layer in autoencoder.encoder_layers:
        layer.trainable = False
    
    # Copy decoder weights to generator for better initialization if needed
    if ae_epochs > 0:
        print("\nCopying decoder weights to generator for initialization...")
        # Manual copy of weights from decoder to generator
        for i, (d_layer, g_layer) in enumerate(zip(
            [l for l in autoencoder.decoder_layers if isinstance(l, tf.keras.layers.Dense)],
            [l for l in generator.layers_list if isinstance(l, tf.keras.layers.Dense)]
        )):
            if d_layer.get_weights()[0].shape == g_layer.get_weights()[0].shape:
                g_layer.set_weights(d_layer.get_weights())
                print(f"Copied weights from {d_layer.name} to {g_layer.name}")
    
    # Phase 2: Train GAN with frozen encoder
    print("\nPhase 2: Training GAN with frozen encoder...")
    
    # Initialize metrics history dictionary to track training progress
    metrics_history = {
        'generator_loss': [],
        'critic_loss': [],
        'wasserstein_distance': [],
        'gradient_penalty': [],
        'reconstruction_loss': [],
        'critic_accuracy': [],
        'generator_fool_rate': [],
        'generator_grad_norm': [],
        'generator_var_count': []
    }
    
    for epoch in range(gan_epochs):
        start_time = time.time()
        
        # Generate batches
        batch_pulses, batch_params = generate_batches(batch_size, input_shape, n_list, params)
        batch_pulses_reshaped = np.reshape(batch_pulses, (batch_size, input_shape[0] * input_shape[1], input_shape[2]))
        
        # Train GAN
        metrics = trainer.train_step(batch_pulses_reshaped, batch_params)
        
        # Log metrics
        with summary_writer.as_default():
            tf.summary.scalar('gan_generator_loss', metrics['generator_loss'], step=epoch + ae_epochs)
            tf.summary.scalar('gan_critic_loss', metrics['critic_loss'], step=epoch + ae_epochs)
            tf.summary.scalar('gan_wasserstein_distance', metrics['wasserstein_distance'], step=epoch + ae_epochs)
            tf.summary.scalar('gan_gradient_penalty', metrics['gradient_penalty'], step=epoch + ae_epochs)
            tf.summary.scalar('gan_critic_accuracy', metrics['critic_accuracy'], step=epoch + ae_epochs)
            tf.summary.scalar('gan_generator_fool_rate', metrics['generator_fool_rate'], step=epoch + ae_epochs)
            tf.summary.scalar('gan_generator_grad_norm', metrics['generator_grad_norm'], step=epoch + ae_epochs)
            tf.summary.scalar('gan_generator_var_count', metrics['generator_var_count'], step=epoch + ae_epochs)
        
        # Print progress
        end_time = time.time()
        print(f"Epoch {epoch+1}/{gan_epochs} - {end_time - start_time:.2f}s")
        print(f"  Generator Loss        | {metrics['generator_loss']:.4f}")
        print(f"  Reconstruction Loss   | {metrics['reconstruction_loss']:.4f}")
        print(f"  Discriminator Loss    | {metrics['critic_loss']:.4f}")
        print(f"  Wasserstein Distance  | {metrics['wasserstein_distance']:.4f}")
        print(f"  Gradient Penalty      | {metrics['gradient_penalty']:.4f}")
        print(f"  Discriminator Accuracy| {metrics['critic_accuracy']:.4f}")
        # print(f"  Generator Fool Rate   | {metrics['generator_fool_rate']:.4f}")
        # print(f"  Generator Grad Norm   | {metrics['generator_grad_norm']:.4f}")
        # print(f"  Generator Var Count   | {int(metrics['generator_var_count'])}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 100 == 0 or epoch == gan_epochs - 1:
            example_pulses = generate_batches(1, input_shape, n_list, params)
            # ae_manager.save()
            gen_manager.save()
            critic_manager.save()
            generate_and_save_sample(
                autoencoder=autoencoder,
                generator=generator,
                input_data=example_pulses,
                params=params,
                epoch=epoch+1 + ae_epochs,
                sample_dir=sample_dir
            )

        for key in metrics_history.keys():
            if key in metrics:
                metrics_history[key].append(metrics[key])

    if gan_epochs > 0:
        plot_path = os.path.join(sample_dir, "gan_training_metrics.png")
        plot_training_metrics(metrics_history, save_path=plot_path)

    example_pulses = generate_batches(1, input_shape, n_list, params)
    generate_and_save_sample(
        autoencoder=autoencoder,
        generator=generator,
        input_data=example_pulses,
        params=params,
        epoch=0,
        sample_dir=sample_dir,
        gif=True
    )
    
    return autoencoder, generator, discriminator


def generate_and_save_sample(autoencoder, generator, input_data, params, epoch, sample_dir, gif=False):
    """
    Generate samples from the model and save them as images using 3D visualization
    """
    input_reshaped = np.reshape(input_data[0], (1, params.R * params.C, params.T))
    encoded = autoencoder.encode(input_reshaped)
    generated = generator(encoded)
    decoded = autoencoder.decode(encoded)

    input_3d = input_data[0][0]
    generated_3d = tf.reshape(generated, (params.R, params.C, params.T)).numpy()
    decoded_3d = tf.reshape(decoded, (params.R, params.C, params.T)).numpy()
    # error_3d = np.abs(input_3d - decoded_3d)
    
    # Create filename
    real_path = os.path.join(sample_dir, f"real_epoch_{epoch}.png")
    generated_path = os.path.join(sample_dir, f"generated_epoch_{epoch}.png")
    decoded_path = os.path.join(sample_dir, f"decoded_epoch_{epoch}.png")
    # error_path = os.path.join(sample_dir, f"error_epoch_{epoch}.png")
    
    # Plot real data
    fig_real = plot_3d_scatter_with_profiles(
        event=input_3d,
        threshold=0.1,
        title=f"Real Data (Epoch {epoch})",
        figsize=(12, 10),
        return_fig=True,
        t_group_size=10
    )
    
    # Save figure and close
    fig_real.savefig(real_path)
    plt.close(fig_real)
    
    # Plot generated data
    fig_generated = plot_3d_scatter_with_profiles(
        event=generated_3d,
        threshold=0.1,
        title=f"Generated Data (Epoch {epoch})",
        figsize=(12, 10),
        return_fig=True,
        t_group_size=10
    )

    # Save figure and close
    fig_generated.savefig(generated_path)
    plt.close(fig_generated)

    fig_decoded = plot_3d_scatter_with_profiles(
        event=decoded_3d,
        threshold=0.1,
        title=f"Decoded Data (Epoch {epoch})",
        figsize=(12, 10),
        return_fig=True,
        t_group_size=10
    )

    fig_decoded.savefig(decoded_path)
    plt.close(fig_decoded)

    if gif:
        concat_events = np.stack([input_3d, generated_3d, decoded_3d], axis=0)
        plot_events(concat_events, title=f"Epoch {epoch}", subtitles=["Real", "Generated", "Decoded"])


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

    max_N = max(n_list)
    params_data = np.zeros((batch_size, max_N, 4))
    for i in range(batch_size):
        params_data[i, :all_N_values[i], :] = np.concatenate([all_vertex_positions[i], all_electron_counts[i][:, None]], axis=1)
    
    batch_pulses = []
    batch_params = []
    
    for i in range(batch_size):
        pulse, _ = simulate_fixed_vertex_electron(
            vertex_positions=all_vertex_positions[i],
            electron_counts=all_electron_counts[i],
            params=params,
            seed=(int(time.time() * 1000) + i) % (2 ** 21)
        )
        
        batch_pulses.append(pulse)
    
    return np.array(batch_pulses), params_data


if __name__ == "__main__":
    # tf.config.run_functions_eagerly(True)  # For debugging
    params = Params()
    
    # Train the model with WGAN architecture
    autoencoder, generator, discriminator = train_dense_gan(
        # Model parameters
        input_shape=(params.R, params.C, params.T),
        latent_dim=128,
        encoder_layer_sizes=[512, 512],
        decoder_layer_sizes=[512, 512],
        generator_layer_sizes=[256, 256],
        critic_layer_sizes=[512, 256],
        
        # Training parameters
        batch_size=8,
        ae_epochs=0,
        gan_epochs=20000,
        feature_matching_weight=0.0,
        reconstruction_weight=0.0,
        generator_weight=1.0,
        critic_steps=2,
        
        # Data parameters
        n_list=[1, 2, 3, 4]
    ) 