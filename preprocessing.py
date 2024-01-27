import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm


def create_sample_adjacency_matrix(num_channels, connectivity=6):
    adjacency_matrix = np.zeros((num_channels, num_channels))

    for i in range(num_channels):
        for j in range(-connectivity // 2, connectivity // 2):
            adjacency_matrix[i, i + j] = 1
            adjacency_matrix[i + j, i] = 1
    
    return adjacency_matrix

def create_simulated_pulses(num_channels, length, std=50, noise=0.1):
    
    pulses = np.zeros((num_channels, length))
    pulse_means = np.random.normal(length / 2, std, num_channels)
    for i in range(num_channels):
        for j in range(length):
            noise_factor = np.random.normal(1, noise)
            magnitude = norm.pdf(pulse_means[i], length / 2, std)
            pulses[i][j] = norm.pdf(j, pulse_means[i], std/np.sqrt(num_channels)) * magnitude * noise_factor

    return pulses

def plot_simulated_pulses(pulses):
    for pulse in pulses:
        plt.plot(pulse)
    
    plt.plot(np.sum(pulses, axis=0))
    plt.show()


def normalize(data):
        if np.linalg.norm(data) == 0:
            print('ALERT', data)
            return None
        else: return data / np.linalg.norm(data)


if __name__ == '__main__':
    pulses = create_simulated_pulses(100, 700)
    print(pulses)
    plot_simulated_pulses(pulses)