import sys
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_training_log(file_path):
    """
    Parse a training log file to extract epoch and loss data.
    """
    epochs = []
    reconstruction_losses = []
    generator_losses = []
    discriminator_losses = []
    discriminator_accuracies = []
    generator_fool_rates = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # Match epoch line
            epoch_match = re.match(r'Epoch (\d+)/\d+', line)
            if epoch_match:
                epochs.append(int(epoch_match.group(1)))
                
            # Match reconstruction loss line
            recon_match = re.match(r'\s+Reconstruction Loss\s+\|\s+([\d\.]+)', line)
            if recon_match:
                reconstruction_losses.append(float(recon_match.group(1)))
                
            # Match generator loss line
            gen_match = re.match(r'\s+Generator Loss\s+\|\s+([-\d\.]+)', line)
            if gen_match:
                generator_losses.append(float(gen_match.group(1)))
                
            # Match discriminator loss line
            disc_match = re.match(r'\s+Discriminator Loss\s+\|\s+([-\d\.]+)', line)
            if disc_match:
                discriminator_losses.append(float(disc_match.group(1)))
                
            # Match discriminator accuracy line
            acc_match = re.match(r'\s+Discriminator Accuracy\|\s+([\d\.]+)', line)
            if acc_match:
                discriminator_accuracies.append(float(acc_match.group(1)))
                
            # Match generator fool rate line
            fool_match = re.match(r'\s+Generator Fool Rate\s+\|\s+([\d\.]+)', line)
            if fool_match:
                generator_fool_rates.append(float(fool_match.group(1)))
    
    return {
        'epochs': epochs,
        'reconstruction_losses': reconstruction_losses,
        'generator_losses': generator_losses,
        'discriminator_losses': discriminator_losses,
        'discriminator_accuracies': discriminator_accuracies,
        'generator_fool_rates': generator_fool_rates
    }

def exponential_moving_average(data, alpha=0.1):
    """
    Calculate exponential moving average of data.
    """
    ema = [data[0]]  # Start with the first data point
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[i-1])
    return ema

def plot_losses(data):
    """
    Create plots of various losses and metrics over epochs.
    """
    epochs = data['epochs']
    
    # Plot reconstruction loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data['reconstruction_losses'], 'b-', alpha=0.5, linewidth=1, label='Raw Loss')
    if len(data['reconstruction_losses']) > 1:
        ema = exponential_moving_average(data['reconstruction_losses'])
        plt.plot(epochs, ema, 'b-', linewidth=2, label='EMA')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('reconstruction_loss.png', dpi=300)
    plt.close()
    
    # Plot generator and discriminator losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data['generator_losses'], 'g-', alpha=0.5, linewidth=1, label='Generator Loss')
    plt.plot(epochs, data['discriminator_losses'], 'r-', alpha=0.5, linewidth=1, label='Discriminator Loss')
    if len(data['generator_losses']) > 1:
        gen_ema = exponential_moving_average(data['generator_losses'])
        plt.plot(epochs, gen_ema, 'g-', linewidth=2, label='Generator EMA')
    if len(data['discriminator_losses']) > 1:
        disc_ema = exponential_moving_average(data['discriminator_losses'])
        plt.plot(epochs, disc_ema, 'r-', linewidth=2, label='Discriminator EMA')
    plt.title('Generator and Discriminator Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('gen_disc_losses.png', dpi=300)
    plt.close()
    
    # Plot discriminator accuracy and generator fool rate
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data['discriminator_accuracies'], 'b-', alpha=0.5, linewidth=1, label='Discriminator Accuracy')
    plt.plot(epochs, data['generator_fool_rates'], 'r-', alpha=0.5, linewidth=1, label='Generator Fool Rate')
    if len(data['discriminator_accuracies']) > 1:
        acc_ema = exponential_moving_average(data['discriminator_accuracies'])
        plt.plot(epochs, acc_ema, 'b-', linewidth=2, label='Accuracy EMA')
    if len(data['generator_fool_rates']) > 1:
        fool_ema = exponential_moving_average(data['generator_fool_rates'])
        plt.plot(epochs, fool_ema, 'r-', linewidth=2, label='Fool Rate EMA')
    plt.title('Discriminator Accuracy and Generator Fool Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('accuracy_fool_rate.png', dpi=300)
    plt.close()

def main():
    if len(sys.argv) != 2:
        print("Usage: python test2.py <training_log_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        data = parse_training_log(file_path)
        
        if not data['epochs']:
            print("No data found in the log file.")
            sys.exit(1)
            
        print(f"Parsed {len(data['epochs'])} epochs of data")
        plot_losses(data)
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
