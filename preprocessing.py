import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import tensorflow as tf

import imageio
from tqdm import tqdm
import os
import datetime

'''
Custom 'print' function
    statements      : Print statements
    end             : End token

    return          : None
'''
def debug_print(statements=[], end='\n'):
    ct = datetime.datetime.now()
    print('[', str(ct)[:19], '] ', sep='', end='')
    for statement in statements:
        print(statement, end=' ')
    print(end=end)


'''
Computes and appends 10/90 value to each pulse sample
    x               : Array of pulses

    return          : Array of pulses with concatenate 10/90 value
'''
def add_1090(x):
        x = x[:, :, 0]
        cdfs = tf.cumsum(x, axis=1)
        time_diffs = np.zeros((x.shape[0], 1))
        for i, cdf in tqdm(enumerate(cdfs)):
            cdf = cdf / cdf[-1]

            time = np.arange(-3500, 3500, 10)  # ns

            # Find the 10% and 90% indices
            idx_10 = np.searchsorted(cdf, 0.1)
            idx_90 = np.searchsorted(cdf, 0.9)

            # Calculate the time difference
            time_diff = time[idx_90] - time[idx_10]
            time_diffs[i] = time_diff
        
        return np.concatenate([x, time_diffs], axis=1)


'''
Adds variable time jitter to pulses
    pulses          : Array of pulses
    t               : Amount of jitter (pm t/2)

    return          : Jittered pulses
'''
def jitter_pulses(pulses, t=50):
    debug_print(['jittering pulses'])
    jitter_pulses = []
    for pulse in tqdm(pulses):
        pulse = np.squeeze(pulse)
        jitter = int(np.random.random() * t - t / 2)
        if jitter < 0: jitter_pulse = np.concatenate([np.zeros(-jitter), pulse[:jitter]], axis=-1)
        else: jitter_pulse = np.concatenate([pulse[jitter:], np.zeros(jitter)], axis=-1)
        jitter_pulses.append(np.expand_dims(jitter_pulse, axis=-1))
    
    return np.expand_dims(np.concatenate(jitter_pulses, axis=-1).T, axis=-1)


'''
Slide pulse by some amount t
'''
def slide_pulse(pulse, t):
    pulse = np.squeeze(pulse)
    if t < 0: pulse = np.concatenate([np.zeros(-t), pulse[:t]], axis=-1)
    else: pulse = np.concatenate([pulse[t:], np.zeros(t)], axis=-1)
    return np.expand_dims(pulse, axis=-1)

'''
Take fast fourier decomposition of pulse data
    signal          : Sample pulse of data

    returns         : Phase shifted fourier decomposition data
'''
def fourier_decomposition(signal, plot=True):
    fft = np.fft.fft(signal)
    n = np.arange(signal.shape[0]) * 0.1
    if plot:
        plt.subplot(2, 2, 1)
        plt.title('Example event')
        plt.plot(signal)
        plt.xlabel('Time')

        plt.subplot(2, 2, 2)
        plt.title('Fourier Decomposition')
        plt.xlabel('Freq (GHz)')
        plt.ylabel('FFT Amplitude')
        plt.stem(n, np.abs(fft), 'b', markerfmt=' ', basefmt='-b')

        plt.subplot(2, 2, 3)
        plt.title('Reconstruction')
        plt.xlabel('Time')
        plt.plot(np.fft.ifft(fft))

        plt.subplot(2, 2, 4)
        plt.title('Reconstruction (Centered waves)')
        plt.xlabel('Time')
        plt.plot(np.fft.ifft(np.abs(fft)))
    
    return np.abs(fft)


'''
Shifts the relative representation of delta mu categories in data
    X               : X data (pulses)
    Y               : Y data (delta mu)

    returns         : Distribution-shifted (X, Y)
'''
def shift_distribution(X, Y):
    def sigmoid(x): return 1/(1 + np.exp(-x))
    def func(x): return (0.2 + sigmoid(5 - x)) / 1.2 * X.shape[0] / 20

    X_list, Y_list = [], []

    for i in range(20):
        start_index = int(i * X.shape[0] / 20)
        end_index = int(start_index + func(i))
        # print(i, start_index, end_index)
        X_list.append(X[start_index:end_index])
        Y_list.append(Y[start_index:end_index])
    
    X, Y = np.concatenate(X_list), np.concatenate(Y_list)
    return X, Y

'''
*unused*
'''
def create_sample_adjacency_matrix(num_channels, connectivity=6):
    adjacency_matrix = np.zeros((num_channels, num_channels))

    for i in range(num_channels):
        for j in range(-connectivity // 2, connectivity // 2):
            adjacency_matrix[i, i + j] = 1
            adjacency_matrix[i + j, i] = 1
    
    return adjacency_matrix

'''
*unused*
'''
def create_simulated_pulses(num_channels, length, std=50, noise=0.1):
    
    pulses = np.zeros((num_channels, length))
    pulse_means = np.random.normal(length / 2, std, num_channels)
    for i in range(num_channels):
        for j in range(length):
            noise_factor = np.random.normal(1, noise)
            magnitude = norm.pdf(pulse_means[i], length / 2, std)
            pulses[i][j] = norm.pdf(j, pulse_means[i], std/np.sqrt(num_channels)) * magnitude * noise_factor

    return pulses

'''
*unused*
'''
def plot_simulated_pulses(pulses):
    for pulse in pulses:
        plt.plot(pulse)
    
    plt.plot(np.sum(pulses, axis=0))
    plt.show()

'''
Normalizes set of data
'''
def normalize(data):
        if np.linalg.norm(data) == 0:
            print('ALERT', data)
            return None
        else: return data / np.linalg.norm(data)

'''
TF data generator
'''
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


def convert_files_to_gif(directory, name):
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(directory, filename)
            images.append(imageio.imread(file_path))

    with imageio.get_writer(f'{directory}{name}', mode='I') as writer:
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image = imageio.imread(directory + filename)
                writer.append_data(image)
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename != name:
            os.remove(file_path)