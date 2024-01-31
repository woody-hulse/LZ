import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import tensorflow as tf


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


class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size, input_size=(700, ), shuffle=True, add_noise=False):
        self.df = df.copy()
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        #self.add_noise = add_noise
        self.augment = add_noise
        self.n = len(self.df)
        self.jitter_amounts = [0.0, 0.01, 0.05, 0.1, 1.0]
        self.jitter_index = 0

    def jitter(self, pulse, jitter_amount=0.01):
        # Gaussian noise parameters
        mean = 0

        std_dev = self.jitter_amounts[self.jitter_index]  # Use current jitter
        return pulse + np.random.normal(mean, std_dev, len(pulse))

    def time_shift(self, pulse):
        sample_shift_amounts = np.array([-2, -1, 0, 1, 2])
        sample_shift_amount = np.random.choice(sample_shift_amounts)
        shifted_pulse = np.roll(pulse, sample_shift_amount)
        if sample_shift_amount > 0: # shift right
            shifted_pulse[:sample_shift_amount] = 0
        elif sample_shift_amount < 0: # shift left
            shifted_pulse[sample_shift_amount:] = 0
        return shifted_pulse

    def augment_data(self, pulse):
        pulse = self.jitter(pulse)
        pulse = self.time_shift(pulse)
        return pulse
    
    def __get_data(self, df_batch):

        training_vars = ["DSdata"]
        target_var = ["UL_values"]

        X_batch = np.array(df_batch[training_vars].values.tolist())
        y_batch = np.array(
            [value[1] if len(value) > 1 and row_truth == 0 else 0.0 for value, row_truth in zip(df_batch["UL_values"].values.tolist(), 
                                                                                                df_batch['truth'])])
        
        X_batch = X_batch.reshape(len(X_batch), -1, 1)

        if self.augment:
            augmented_X_batch = np.array([self.augment_data(pulse[:, 0]) for pulse in X_batch])
            augmented_X_batch = augmented_X_batch.reshape(len(augmented_X_batch), -1, 1)

            # Concatenate original and augmented X_batch along the 0 axis (batch dimension)
            X_batch = np.concatenate((X_batch, augmented_X_batch), axis=0)
            
            # Duplicate y_batch to match the new size of X_batch
            y_batch = np.concatenate((y_batch, y_batch), axis=0)

            #print(y_batch[:10])
            # Count non-zero entries
            #non_zero_count = np.count_nonzero(y_batch)
            #print("Number of non-zero entries:", non_zero_count)
            return X_batch, y_batch    

        else:
            # Count non-zero entries
            #non_zero_count = np.count_nonzero(y_batch)
            #print("Number of non-zero entries:", non_zero_count)
            return X_batch, y_batch
        
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __getitem__(self, index):
        df_batch = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(df_batch)
        
        return X, y

    def __len__(self):
        return self.n // self.batch_size


if __name__ == '__main__':
    pulses = create_simulated_pulses(100, 700)
    print(pulses)
    plot_simulated_pulses(pulses)