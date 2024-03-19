import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import sys
import copy

import matplotlib.pyplot as plt
import seaborn as sns

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

# import importlib
# importlib.reload(regression_models)

DSS_NAME = '../dSSdMS/dSS_230918_gaussgas_700sample_area7000_1e5events_random_centered.npz'
DMS_NAME = '../dSSdMS/dMS_231202_gaussgas_700sample_area7000_areafrac0o5_deltamuinterval50ns_5000each_5e4events_random_centered_above1000ns_batch00.npz'
DMS_NAMES = [
    '/Users/woodyhulse/Documents/lz/dSSdMS/dMS_240306_gaussgas_700sample_148electrons_randomareafrac_deltamuinterval50ns_5000each_1e5events_random_jitter100ns_batch00.npz',
    # '/Users/woodyhulse/Documents/lz/dSSdMS/dMS_240306_gaussgas_700sample_148electrons_randomareafrac_deltamuinterval50ns_5000each_1e5events_random_jitter100ns_batch01.npz',
    # '/Users/woodyhulse/Documents/lz/dSSdMS/dMS_240306_gaussgas_700sample_148electrons_randomareafrac_deltamuinterval50ns_5000each_1e5events_random_jitter100ns_batch02.npz'
]

# '/Users/woodyhulse/Documents/lz/dSSdMS/dMS_231011_gaussgas_700sample_area7000_areafrac0o5_deltamuinterval50ns_5000each_1e5events_random_centered_batch10.npz'

MODEL_SAVE_PATH = 'saved_models/'

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
def train(model, X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, compile=False, summary=False, callbacks=False, learning_rate=0.001, metrics=[]):
    debug_print(['training', model.name])

    if compile:
        model.compile(optimizer=model.optimizer, loss=model.loss, metrics=metrics)
        model.build(X_train.shape)
    if summary:
        model.summary()

    c = []
    if callbacks:
        tf.keras.backend.set_value(model.optimizer.learning_rate, learning_rate)
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
    
    return history

'''
Perform classification-based tasks and experiments
'''
def classification():
    num_samples = 300000

    dSS = DS(DSS_NAME)
    dMS = DS(DMS_NAME)

    debug_print(['preprocessing data'])
    data_indices = np.array([i for i in range(len(dSS.ULvalues) + len(dMS.ULvalues))])
    np.random.shuffle(data_indices)
    X = np.concatenate([dSS.DSdata, dMS.DSdata], axis=0)[data_indices][:num_samples]
    X = np.concatenate([np.expand_dims(normalize(dist), 0) for dist in X], axis=0)

    X_avg = normalize(np.average(X, axis=0))
    x = np.linspace(0, X.shape[1], X.shape[1])
    params, cov = curve_fit(gaussian, x, X_avg)
    X_dist = gaussian(x, *params)
    X_dev = np.concatenate([np.expand_dims(X[i] / X_dist, axis=0) for i in range(X.shape[0])], axis=0)
    X_diff = np.concatenate([np.expand_dims(X[i] - X_dist, axis=0) for i in range(X.shape[0])], axis=0)

    X = np.expand_dims(X, axis=-1)
    Y = np.concatenate([
        [0 for _ in range(len(dSS.ULvalues))],
        [1 for _ in range(len(dMS.ULvalues))]
        ], axis=0)[data_indices][:num_samples]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # X_dev_train, X_dev_test, Y_train, Y_test = train_test_split(X_dev, Y, test_size=0.2, random_state=42)
    # X_diff_train, X_diff_test, Y_train, Y_test = train_test_split(X_diff, Y, test_size=0.2, random_state=42)

    debug_print(['     X_train:', X_train.shape])
    debug_print(['     Y_train:', Y_train.shape])

    # test_model = build_vgg(length=700, name='vgg', width=16) # MLPModel(output_size=1)
    baseline_model = MLPModel(input_size=700, classification=True)
    test_model = HybridModel(input_size=700, classification=True)
    test_model2 = ConvNoAttentionModel(input_size=700,  output_size=1, classification=True)

    # train(baseline_model, X_train, Y_train, epochs=50, batch_size=16, callbacks=False)
    train(test_model2, X_train, Y_train, epochs=50, batch_size=128, callbacks=False)
    # train(test_model2, X_dev_train, Y_train, epochs=25, batch_size=16, compile=True)
    # train(test_model3, X_diff_train, Y_train, epochs=25, batch_size=16, compile=True)

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
    np.random.seed(42)
    num_samples = 100000

    def normalize(data):
        if np.linalg.norm(data) == 0:
            print('ALERT', data)
            return None
        else: return data / np.linalg.norm(data)

    debug_print(['preprocessing data'])
    
    X, Y, areafrac = concat_data(DMS_NAMES)
    # X, Y = shift_distribution(X, Y)
    # plot_distribution(Y)
    data_indices = np.array([i for i in range(len(X))])
    np.random.shuffle(data_indices)
    X = X[data_indices][:num_samples]
    X = np.concatenate([np.expand_dims(normalize(dist), 0) for dist in X], axis=0)
    Y = Y[data_indices][:num_samples]
    areafrac = areafrac[data_indices][:num_samples]

    # X = jitter_pulses(X, t=20)
    
    X_avg = normalize(np.average(X, axis=0))
    x = np.linspace(0, X.shape[1], X.shape[1])
    params, cov = curve_fit(gaussian, x, X_avg)
    X_dist = gaussian(x, *params)
    X_dist = np.expand_dims(X_dist, axis=-1)
    '''
    X_dev = np.concatenate([np.expand_dims(X[i] / X_dist, axis=0) for i in range(X.shape[0])], axis=0)
    X_diff = np.concatenate([np.expand_dims(X[i] - X_dist, axis=0) for i in range(X.shape[0])], axis=0)
    '''

    '''
    X_fft = np.concatenate([np.expand_dims(fourier_decomposition(x, plot=False), axis=0) for x in tqdm(X)])
    X_fft = np.expand_dims(X_fft, axis=-1)

    X_dev_train, X_dev_test, Y_train, Y_test = train_test_split(X_dev, Y, test_size=0.2, random_state=42)
    X_diff_train, X_diff_test, Y_train, Y_test = train_test_split(X_diff, Y, test_size=0.2, random_state=42)
    '''

    X = np.expand_dims(X, axis=-1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # X_fft_train, X_fft_test, Y_train, Y_test = train_test_split(X_fft, Y, test_size=0.2, random_state=42)
    # X_1090_train = add_1090(X_train)
    # X_1090_test = add_1090(X_test)

    debug_print(['     X_train:', X_train.shape])
    debug_print(['     Y_train:', Y_train.shape])

    '''
    Experiments
    '''

    compare_latent_dim_compression([1, 2, 4, 8, 16, 32, 64, 128, 256], X_train, Y_train, X_test, Y_test)


    # tuner = keras_tuner.RandomSearch(tuner_model, objective='val_loss', max_trials=100)
    # tuner.search(np.squeeze(X_train), Y_train, epochs=35, batch_size=128, validation_data=(np.squeeze(X_test), Y_test))

    # fig = plot_parameter_performance(['untitled_project/', '../untitled_project_2/', '../untitled_project_4/'], title='Number of Parameters vs. Training Performance [val MAE] [3/9/24]')

    # save_interactive_plot(fig)
    # load_interactive_plot()

def channel_classification():
    num_samples = 1000
    num_channels = 100

    data = np.array([create_simulated_pulses(num_channels, 700) for _ in range(num_samples)])
    example_labels = np.zeros((num_samples,))
    spatial_data, temporal_data = PODMLP.POD(data)

    train_data, test_data, train_labels, test_lables = train_test_split([data, example_labels])

    sample_adjacency_matrix = create_sample_adjacency_matrix(num_channels, connectivity=6)

    print(sample_adjacency_matrix)

    test_gnn = BaselineGNN(num_channels=num_channels)


def main():
    regression()


if __name__ == '__main__':
    # os.system('clear')
    main()