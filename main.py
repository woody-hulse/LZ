import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import sys
import copy

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='whitegrid',rc={'font.family': 'sans-serif','font.serif':'Times'})

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.stats import norm
import matplotlib.cm as cm
from preprocessing import *
# from models import *
from regression_models import *
from channel_models import *
from vgg import *
from experiments import *
from autoencoder import *
from pulse import *

DATE = ' [5-15-24]'

'''
Data filepaths
'''
# Typical single scatter (SS) events (100k)
DS_FILE = '../dSSdMS/dSS_230918_gaussgas_700sample_area7000_1e5events_random_centered.npz'
# Typical multi-scatter (MS) events (100k)
DMS_FILE = '../dSSdMS/dMS_231202_gaussgas_700sample_area7000_areafrac0o5_deltamuinterval50ns_5000each_5e4events_random_centered_above1000ns_batch00.npz'
# Three batches of MS events (100k x 3)
DMS_FILES = [
    '../dSSdMS/dMS_240306_gaussgas_700sample_148electrons_randomareafrac_deltamuinterval50ns_5000each_1e5events_random_jitter100ns_batch00.npz',
    '../dSSdMS/dMS_240306_gaussgas_700sample_148electrons_randomareafrac_deltamuinterval50ns_5000each_1e5events_random_jitter100ns_batch01.npz',
    '../dSSdMS/dMS_240306_gaussgas_700sample_148electrons_randomareafrac_deltamuinterval50ns_5000each_1e5events_random_jitter100ns_batch02.npz'
]
# Single-photon SS events (100k)
DSS_SIMPLE_FILE = '/Users/woodyhulse/Documents/lz/dSSdMS/dSS_2400419_gaussgass_700samplearea7000_areafrac0o5_1.0e+05events_random_centered_withEAT_1e.npz'
# MS events with corresponding electron arrival times
DMS_AT_FILE = '../dSSdMS/dSS_2400417_gaussgass_700samplearea7000_areafrac0o5_1.0e+05events_random_centered_channel_withEAT_10e.npz'
# MS events with channel-level resolution and corresponding electron arrival times
DMS_AT_CHANNEL_FILE = '../dSSdMS/dSS_2400417_gaussgass_700samplearea7000_areafrac0o5_1.0e+05events_random_centered_channel_withEAT.npz'
# MS events with channel-level resolution
DMS_CHANNEL_FILE = '../dSSdMS/dSS_2400417_gaussgass_700samplearea7000_areafrac0o5_1.0e+05events_random_centered_channel_withEAT.npz'
# SS events with associated (binned) photon arrivals
# DSS_CHANNEL_PHOTON_FILE = '../dSSdMS/dSS_2400501_gaussgass_700samplearea7000_areafrac0o5_1.0e+04events_random_centered.npz'
# DSS_CHANNEL_PHOTON_FILE = '../dSSdMS/dSS_2400913_gaussgass_700samplearea7000_areafrac0o5_1.0e+04events_random_centered.npz'
DSS_CHANNEL_PHOTON_FILE = '../dSSdMS/dSS_2400914_gaussgass_700samplearea7000_areafrac0o5_1.0e+05events_random_centered.npz'

'''
Training function for tensorflow model
    model           : Tensorflow model
    X_train         : X training data (pulses)
    y_train         : Y training data (delta mu)
    epochs          : Number of training loops
    batch_size      : Number of samples per batch
    validation_split: Proportion of sample data to be dedicated as validation
    compile         : Compile the model before training
    summary         : Provide a summary of model before training
    callbacks       : Add EarlyStopping, ReduceLROnPlateau to training cycle
    learning_rate   : Designate learning rate for training start
    metrics         : Training evaluation metrics

    return          : Model training history (metrics + losses)
'''
def train(model, X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, compile=False, summary=False, callbacks=False, learning_rate=0.001, metrics=[], plot_history=False):
    debug_print(['training', model.name])

    if compile:
        model.compile(optimizer=model.optimizer, loss=model.loss, metrics=metrics)
        model.build(X_train.shape)
    if summary:
        model.summary()

    c = []
    if callbacks:
        # tf.keras.backend.set_value(model.optimizer.learning_rate, learning_rate)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
        c.append(tf.keras.callbacks.EarlyStopping(patience=20))
        c.append(reduce_lr)

    history = model.fit(
        X_train,
        y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=validation_split, 
        verbose=1,
        callbacks=c
        )
    
    if plot_history:
        # plt.plot(history.history['loss'], label='Loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()
    
    return history


'''
Plot the distribution of pulse delta mu labels
    Y           : Delta mu labels

    return      : None
'''
def plot_distribution(Y):
    counts, bins = np.histogram(Y)
    plt.stairs(counts / Y.shape[0], bins, color='orange', fill=True)
    plt.xlabel('Δμ')
    plt.title('Distribution of model training data')
    plt.show()


'''
Perform regression-based tasks and experiments
'''
def regression():

    '''
    Data preprocessing
    '''
    np.random.seed(42)
    num_events = 50000

    def normalize(data):
        if np.linalg.norm(data) == 0:
            print('ALERT', data)
            return None
        else: return data / np.linalg.norm(data)

    debug_print(['preprocessing data'])

    X     = np.array([None for _ in range(int(1e6))]) # pulse
    Y     = np.array([None for _ in range(int(1e6))]) # delta mu
    XC    = np.array([None for _ in range(int(1e6))]) # pulse (channel-level)
    PXC   = np.array([None for _ in range(int(1e6))]) # discrete photon arrival (channel-level)
    AF    = np.array([None for _ in range(int(1e6))]) # area fraction
    AT    = np.array([None for _ in range(int(1e6))]) # arrival time

    X, XC, PXC, Y, AT = load_pulse_dataset(DSS_CHANNEL_PHOTON_FILE)

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # X_XC_PXC_Y_AT_train, X_XC_PXC_Y_AT_test = train_test_split_all([X, XC, PXC, Y, AT], test_size=0.2)
    
    event_indices = np.array([i for i in range(min(X.shape[0], num_events))])
    np.random.shuffle(event_indices)
    X       = X[event_indices][:num_events]     # Summed pulse signal
    XC      = XC[event_indices][:num_events]    # Channel pulse signal
    PXC     = PXC[event_indices][:num_events]   # Photon count channel pulse signal
    Y       = Y[event_indices][:num_events]     # Delta mu for MS, 0 for SS
    AT      = AT[event_indices][:num_events]    # Electron arrival tiimes
    AF      = AF[event_indices][:num_events]    # Area fraction of MS pulse

    num_samples = 1000000
    sample_indices = np.array([i for i in range(num_samples)])
    np.random.shuffle(sample_indices)
    X_FLAT = np.reshape(X, (-1, 1))
    XC_FLAT = np.reshape(XC, (-1, 1))
    PXC_FLAT = np.reshape(PXC, (-1, 1))
    PX_FLAT = np.reshape(np.sum(PXC, axis=(1, 2)), (-1, 1))

    '''
    Experiments
    '''
    # plot_hit_pattern(XC)
    # graph_electron_arrival_prediction(XC, AT[:, :, -1], epochs=50)
    graph_channel_electron_arrival_prediction(XC, AT, epochs=25)

    # test_graph_network()
    

def main():
    regression()


if __name__ == '__main__':
    # os.system('clear')
    main()