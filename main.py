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
DMS_FILE = '../dSSdMS/dMS_231011_gaussgas_700sample_area7000_areafrac0o5_deltamuinterval50ns_5000each_1e5events_random_centered_batch10.npz'
# Continuous multi-scatter (MS) events (100k)
DMS_CONTINUOUS_FILE = '../dSSdMS/dMS_240926_gaussgas_700sample_area7000_areafrac0o5_randomdeltamu_1e5events_centered_batch00.npz'

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
# DSS_CHANNEL_PHOTON_FILE = '../dSSdMS/dSS_2400920_gaussgass_700samplearea7000_areafrac0o5_5.0e+04events_random_centered.npz' # SIMPLE
# DSS_CHANNEL_PHOTON_FILE = '../dSSdMS/dSS_2400917_gaussgass_700samplearea7000_areafrac0o5_5.0e+04events_random_centered.npz'
# DSS_CHANNEL_PHOTON_FILE = '../dSSdMS/dSS_2400921_gaussgass_700samplearea7000_areafrac0o5_5.0e+04events_random_centered.npz' # NORMAL
DSS_CHANNEL_PHOTON_FILE = '../dSSdMS/dSS_2400928_gaussgass_700samplearea7000_areafrac0o5_5.0e+04events_random_centered.npz' # RANDOM

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
    if summary:
        model.summary()

    c = []
    if callbacks:
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
        plt.plot(history.history['loss'], label='Training loss')
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
    num_events = 100000

    def normalize(data):
        if np.linalg.norm(data) == 0:
            print('ALERT', data)
            return None
        else: return data / np.linalg.norm(data)

    debug_print(['preprocessing data'])

    X     = np.array([None] * int(1e6)) # pulse
    Y     = np.array([None] * int(1e6)) # delta mu
    XC    = np.array([None] * int(1e6)) # pulse (channel-level)
    PXC   = np.array([None] * int(1e6)) # discrete photon arrival (channel-level)
    AF    = np.array([None] * int(1e6)) # area fraction
    AT    = np.array([None] * int(1e6)) # arrival time

    # X, XC, PXC, Y, AT = load_pulse_dataset(DMS_FILE)
    X, Y = load_pulse_dataset_old(DMS_FILE)
    X_continuous, Y_continuous = load_pulse_dataset_old(DMS_CONTINUOUS_FILE)
    X_binned = np.array([X[i * 5000 : (i + 1) * 5000] for i in range(20)])
    Y_binned = np.array([Y[i * 5000 : (i + 1) * 5000] for i in range(20)])

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

    X_continuous = X_continuous[event_indices][:num_events]
    Y_continuous = Y_continuous[event_indices][:num_events]

    num_samples = 1000000
    sample_indices = np.arange(num_samples)
    np.random.shuffle(sample_indices)
    X_FLAT = np.reshape(X, (-1, 1))
    XC_FLAT = np.reshape(XC, (-1, 1))
    # PXC_FLAT = np.reshape(PXC, (-1, 1))
    # PX_FLAT = np.reshape(np.sum(PXC, axis=(1, 2)), (-1, 1))
    '''
    Experiments
    '''
    # print(np.sum(XC[0]), np.sum(AT[0]))
    # print(np.sum(np.abs(XC[
    # 0] - AT[0])), np.sum(XC[0] - AT[0]))
    # plot_hit_pattern(XC[0])
    # graph_electron_arrival_prediction(XC, AT[:, :, -1], epochs=50)
    # graph_channel_electron_arrival_prediction(XC, AT, epochs=0)
    # graph_electron_arrival_prediction(np.array(PXC, np.float32), AT, epochs=30)
    # PXC_sum = np.sum(PXC, axis=(1, 2))
    # XC_sum = np.sum(XC, axis=(1, 2))
    
    # graph_electron_arrival_prediction(XC, AT, epochs=10)
    # conv_graph_electron_arrival_prediction(XC, AT, epochs=25, dim3=False, savefigs=True)
    # conv_graph_electron_arrival_prediction(XC, AT, epochs=6, dim3=True, savefigs=True)

    # model = MLPModel()
    # model.compile(optimizer=Adam(learning_rate=1e-4), loss=mean_log_error, metrics=[MeanAbsoluteError()])
    # model.build((None, 700))
    # train(model, X, Y, epochs=100, batch_size=512, validation_split=0.2, summary=True, plot_history=False)
    # model.fit(X, Y, epochs=100, batch_size=512, validation_split=0.2, verbose=1)

    # plot_histogram(X_binned, Y_binned, model)


    
    conf_model = load('skew_dist_conf_model')

    # conf_model = MLPNormalDistributionModel()
    # conf_model.build((None, 700))
    conf_model.compile(optimizer=Adam(learning_rate=3e-6), loss=skewnormal_pdf_loss_penalty, metrics=[mu_loss, skew_mu_loss])

    train(conf_model, X_continuous, Y_continuous, epochs=50, batch_size=512, validation_split=0.2, summary=True, plot_history=True)
    # save_model(conf_model, 'pos_dist_conf_model')

    # plot_prediction_percentile_distribution_normal(X_continuous, Y_continuous, conf_model, skewnormal_pdf)

    # plot_prediction_z_distribution_normal(X, Y, conf_model, normal_pdf)

    plot_distribution_histogram(X_binned, Y_binned, conf_model, pdf_func=normal_pdf)
    # plot_distribution_model_examples(X, Y, conf_model, pdf_func=normal_pdf, num_examples=20)
    # plot_pdf_gradient_histogram(X_binned, Y_binned, conf_model)
    


    
    # conf_model = load('skew_dist_conf_model')

    # plot_prediction_percentile_distribution_normal(X, Y, conf_model, skewnormal_pdf)

    # conf_model = MLPDistributionModel()
    # conf_model.build((None, 700))
    # conf_model.compile(optimizer=Adam(learning_rate=5e-7), loss=skewnormal_pdf_loss, metrics=[mu_loss, skew_mu_loss])

    # train(conf_model, X, Y, epochs=100, batch_size=512, validation_split=0.2, summary=True, plot_history=True)
    # save_model(conf_model, 'skew_dist_conf_model')

    # plot_prediction_z_distribution_skewnormal(X, Y, conf_model, skewnormal_pdf)

    # plot_distribution_histogram(X_binned, Y_binned, conf_model, pdf_func=skewnormal_pdf)
    # plot_distribution_model_examples(X, Y, conf_model, pdf_func=skewnormal_pdf, num_examples=20)
    # plot_pdf_gradient_histogram(X_binned, Y_binned, conf_model)
    
    # custom_conf_model = MLPCustomDistributionModel()

    # custom_conf_model = load('custom_conf_model')

    # custom_conf_model.compile(optimizer=Adam(learning_rate=1e-4), loss=tf.keras.losses.SparseCategoricalCrossentropy())
    # train(custom_conf_model, X_continuous, np.array(Y_continuous, dtype=np.int32), epochs=50, batch_size=512, validation_split=0.2, summary=True, plot_history=False)

    # save_model(custom_conf_model, 'custom_conf_model')

    # plot_prediction_percentile_distribution_normal(X_continuous, Y_continuous, custom_conf_model, identity_pdf)

    '''

    custom_conf_model.build((None, 700))
    custom_conf_model.compile(optimizer=Adam(learning_rate=1e-4), loss=tf.keras.losses.SparseCategoricalCrossentropy())
    train(custom_conf_model, X_continuous, np.array(Y_continuous, dtype=np.int32), epochs=100, batch_size=512, validation_split=0.2, summary=True, plot_history=False)

    save_model(custom_conf_model, 'custom_conf_model')

    plot_distribution_histogram(X_binned, Y_binned, custom_conf_model, pdf_func=identity_pdf)
    plot_distribution_model_examples(X, Y, custom_conf_model, pdf_func=identity_pdf, num_examples=20)
    # plot_pdf_gradient_histogram(X_binned, Y_binned, conf_model)
    '''


    # test_graph_network()

    # conf_model = load('bin_conf_model')
    # conf_model = MLPConfidenceModel()
    # conf_model.build((None, 700))
    # conf_model.compile(optimizer=Adam(learning_rate=1e-5), loss=confidence_mae_loss, metrics=[eval_mae_loss, eval_conf])

    # train(conf_model, X, Y, epochs=100, batch_size=512, validation_split=0.2, summary=True, plot_history=False)

    # save_model(conf_model, 'bin_conf_model')

    # plot_confidence_gradient_histogram(X_binned, Y_binned, conf_model)
    # plot_confidence_histogram(X_binned, Y_binned, conf_model)
    

def main():
    regression()


if __name__ == '__main__':
    # os.system('clear')
    main()