import os
import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.stats import norm
from scipy.optimize import curve_fit

from preprocessing import *
# from models import *
from regression_models import *
from channel_models import *
from vgg import *

DSS_NAME = '../dSSdMS/dSS_230918_gaussgas_700sample_area7000_1e5events_random_centered.npz'
DMS_NAME = '../dSSdMS/dMS_230918_gaussgas_700sample_area7000_areafrac0o3_deltamuall50ns_5000each_1e5events_random_centered.npz'

MODEL_SAVE_PATH = 'saved_models/'

class DS():
    def __init__(self, DSname):
        self.DSname = DSname
        self.DStype = 'test'

        path = self.DSname
        with np.load(path) as f:
            debug_print(['loading', self.DStype, 'data from', path])
            fkeys = f.files
            self.DSdata = f[fkeys[0]]       # arraysize =（event * 700 samples)
            self.ULvalues = f[fkeys[1]]     # arraysize =（event * 4 underlying parameters)


def debug_print(statements=[], end='\n'):
    ct = datetime.datetime.now()
    print('[', str(ct)[:19], '] ', sep='', end='')
    for statement in statements:
        print(statement, end=' ')
    print(end=end)


def save_model_weights(model, name=None):
    if not name: name = model.name
    debug_print(['saving', name, 'weights to', MODEL_SAVE_PATH])
    model.save_weights(MODEL_SAVE_PATH + name)


def load_model_weights(model, name=None):
    if not name: name = model.name
    debug_print(['loading', name, 'weights from', MODEL_SAVE_PATH + name + '.data'])
    model.load_weights(MODEL_SAVE_PATH + name)



def linearity_plot(model, data=None, delta_mu=50, num_delta_mu=20, num_samples=1000):
    def plot(x, y, err_down, err_up, color='blue', marker='o', title=None):
        plt.plot(x, x, color='black', label='y = x Line')
        plt.errorbar(x, y, yerr=(err_down, err_up), label='\u0394\u03bc Prediction', color=color, marker=marker, linestyle='None')

        plt.xlabel('Actual \u0394\u03bc [ns]')
        plt.ylabel('Median Predicted \u0394\u03bc [ns]')
        plt.title(title)
        plt.legend()
        plt.show()
    
    
    if not data:
        dMS = DS(DMS_NAME)
        data_indices = np.array([i for i in range(len(dMS.ULvalues))])
        np.random.shuffle(data_indices)
        X = dMS.DSdata[data_indices]
        X = np.concatenate([np.expand_dims(normalize(dist), 0) for dist in X], axis=0)
        Y = dMS.ULvalues[data_indices]
        Y = Y[:, 1]
    else:
        X, Y = data
        X = np.squeeze(X)

    counts = np.zeros(num_delta_mu)
    X_delta_mu = np.empty((num_delta_mu, num_samples, 700))

    if '1090' in model.name.split('_'):
        X_delta_mu = np.empty((num_delta_mu, num_samples, 701))
        X = add_1090(np.expand_dims(X, axis=-1))

    for i in tqdm(range(len(X))):
        index = int(Y[i] // delta_mu)
        if counts[index] >= num_samples: continue
        else:
            X_delta_mu[index, int(counts[index]), :] = X[i]
            counts[index] += 1

    predictions = np.zeros((num_delta_mu, num_samples))

    prediction_means = np.zeros(num_delta_mu)
    prediction_left_stds = np.zeros(num_delta_mu)
    prediction_right_stds = np.zeros(num_delta_mu)

    prediction_medians = np.zeros(num_delta_mu)
    prediction_16 = np.zeros(num_delta_mu)
    prediction_84 = np.zeros(num_delta_mu)

    for i in tqdm(range(num_delta_mu)):
        samples = np.expand_dims(X_delta_mu[i], axis=-1)
        predictions[i][:] = np.squeeze(model(samples))
        prediction_means[i] = np.mean(predictions[i])
        prediction_left_stds[i] = np.std(predictions[i][predictions[i] < prediction_means[i]])
        prediction_right_stds[i] = np.std(predictions[i][predictions[i] >= prediction_means[i]])
        
        prediction_medians[i] = np.median(predictions[i])
        hist, bin_edges = np.histogram(predictions[i], bins=100, density=True)
        cumulative_distribution = np.cumsum(hist * np.diff(bin_edges))
        total_area = cumulative_distribution[-1]

        prediction_16[i] = prediction_medians[i] - bin_edges[np.where(cumulative_distribution >= 0.16 * total_area)[0][0]]
        prediction_84[i] = bin_edges[np.where(cumulative_distribution >= 0.84 * total_area)[0][0]] - prediction_medians[i]
        
    plot(np.arange(0, delta_mu * num_delta_mu, delta_mu), 
         prediction_medians, prediction_16, prediction_84, 
         title=model.name + ' model linearity plot')


def gaussian(X, C, sigma,):
    return C*np.exp(-(X-350)**2/(2*sigma**2)) # hard-coded mean

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


def reset_weights(model):
    debug_print(['resetting', model.name, 'weights'])
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Convolution1D, tf.keras.layers.Dense)):
            layer.set_weights([
                tf.keras.initializers.glorot_normal()(layer.weights[0].shape),
                tf.zeros(layer.weights[1].shape)  # Bias
            ])
    return model


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


def small_test(file): # 'SS_vs_MS_dataset.pkl'
    df = pd.read_pickle(file)
    df_shuffled = df[:5000].sample(frac=1)
    # display(df)
    # display(df_shuffled)

    # ex = df_shuffled['DSdata'].iloc
    # print(len(ex[0]), ex[0], ex[1])

    training = df_shuffled[:5000]['DSdata']
    target = df_shuffled[:5000]['truth']

    # display(training)
    # display(target) 

    X = np.array(training.values.tolist())
    X = np.expand_dims(normalize(X), axis=-1)
    y = np.array(target.values.tolist())#.flatten()

    '''
    for i in range(50):
        if y[i] == 0: plt.plot(X[i], color='blue')
        if y[i] == 1: plt.plot(X[i], color='orange')
    plt.show()
    '''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    debug_print(['X_train:', X_train.shape])
    debug_print(['y_train:', y_train.shape])

    test_model = ConvModel2(input_size=X.shape[0])
    baseline_model = BaselineModel().build_model()

    debug_print(['training', baseline_model.name])
    train(baseline_model, X_train, y_train, epochs=50, batch_size=16)

    debug_print(['training', test_model.name])
    train(test_model, X_train, y_train, epochs=50, batch_size=16, compile=True)


def classification():
    num_samples = 5000

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


def jitter_test(models, X, Y, epochs=100, plot_jitter=True):

    debug_print(['running jitter test'])

    sns.set_theme(style="whitegrid")

    jitter = [300]
    X_jitter = []
    for j in jitter:
        x_jitter = jitter_pulses(X, j)
        if plot_jitter:
            for i in range(100):
                plt.plot(x_jitter[i, :, 0])
            plt.title("Plot of random time-translated events: ±" + str(int(j / 2) * 10) + "ns")
            plt.show()
        X_jitter.append(x_jitter)

    losses = []
    for index, model in enumerate(models):
        for i, (x, j) in enumerate(zip(X_jitter, jitter)):
            debug_print(['model', index + 1, '/', len(models), ':', model.name, '- test', i + 1, 'of', len(jitter)])
            # model = reset_weights(model)
            history = train(model, x, Y, epochs=epochs, batch_size=16, validation_split=0.2, compile=False, summary=False, callbacks=False)
            loss = history.history['val_loss']
            losses.append([j, model.name, round(loss[-1], 3)])
            linearity_plot(model, data=(x, Y))
    df = pd.DataFrame(losses, columns=['Jitter', 'Model', 'MAE Loss: ' + str(epochs) + ' epochs'])

    palette = sns.color_palette("rocket_r")
    g = sns.catplot(data=df, kind='bar', x='Model', y='MAE Loss: ' + str(epochs) + ' epochs', hue='Jitter', errorbar='sd', palette=palette, alpha=0.7, height=6)
    g.fig.suptitle('Model sensitivity to temporal variation')
    g.despine(left=True)
    g.set_axis_labels('', 'Loss (MAE) after ' + str(epochs) + ' epochs')
    g.legend.set_title('Variation (\u0394t)')
    plt.show()


    

def compare_history(models, metric, title=None):
    for model in models:
        plt.plot(model.history.history[metric], label=model.name)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(title)
    plt.show()


def shift_distribution(X, Y):
    def sigmoid(x): return 1/(1 + np.exp(-x))
    def func(x): return (1 + sigmoid(10 - x)) / 2 * X.shape[0] / 20

    X_list, Y_list = [], []

    for i in range(20):
        start_index = int(i * X.shape[0] / 20)
        end_index = int(start_index + func(i))
        # print(i, start_index, end_index)
        X_list.append(X[start_index:end_index])
        Y_list.append(Y[start_index:end_index])
    
    X, Y = np.concatenate(X_list), np.concatenate(Y_list)
    return X, Y


def plot_distribution(Y):
    counts, bins = np.histogram(Y)
    plt.stairs(counts / Y.shape[0], bins, color='orange', fill=True)
    plt.xlabel('Δμ')
    plt.title('Distribution of model training data')
    plt.show()


def regression():
    np.random.seed(42)
    num_samples = 100000

    dMS = DS(DMS_NAME)

    def normalize(data):
        if np.linalg.norm(data) == 0:
            print('ALERT', data)
            return None
        else: return data / np.linalg.norm(data)

    debug_print(['preprocessing data'])
    X = dMS.DSdata
    Y = np.array(dMS.ULvalues)[:, 1]
    # X, Y = shift_distribution(X, Y)
    # plot_distribution(Y)
    data_indices = np.array([i for i in range(len(X))])
    np.random.shuffle(data_indices)
    X = X[data_indices][:num_samples]
    X = np.concatenate([np.expand_dims(normalize(dist), 0) for dist in X], axis=0)

    X = jitter_pulses(X, t=300)

    '''
    X_avg = normalize(np.average(X, axis=0))
    x = np.linspace(0, X.shape[1], X.shape[1])
    params, cov = curve_fit(gaussian, x, X_avg)
    X_dist = gaussian(x, *params)
    X_dev = np.concatenate([np.expand_dims(X[i] / X_dist, axis=0) for i in range(X.shape[0])], axis=0)
    X_diff = np.concatenate([np.expand_dims(X[i] - X_dist, axis=0) for i in range(X.shape[0])], axis=0)
    '''

    X_fft = np.concatenate([np.expand_dims(fourier_decomposition(x, plot=False), axis=0) for x in tqdm(X)])
    X = np.expand_dims(X, axis=-1)
    X_fft = np.expand_dims(X_fft, axis=-1)
    Y = Y[data_indices][:num_samples]

    '''
    X_dev_train, X_dev_test, Y_train, Y_test = train_test_split(X_dev, Y, test_size=0.2, random_state=42)
    X_diff_train, X_diff_test, Y_train, Y_test = train_test_split(X_diff, Y, test_size=0.2, random_state=42)
    '''

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_fft_train, X_fft_test, Y_train, Y_test = train_test_split(X_fft, Y, test_size=0.2, random_state=42)
    # X_1090_train = add_1090(X_train)
    # X_1090_test = add_1090(X_test)

    debug_print(['     X_train:', X_train.shape])
    debug_print(['     Y_train:', Y_train.shape])

    ten_layer_conv = ConvModel(input_size=700, name='10_layer_conv')
    two_layer_conv = BaselineConvModel(output_size=1).build_model()
    vgg16_small = build_vgg(length=700, name='vgg16_small', width=16)
    vgg13 = build_vgg(length=700, name='vgg13', width=16)
    vgg16 = build_vgg(length=700, name='vgg16', width=32)
    attention_model = ConvAttentionModel(input_size=700, output_size=1, classification=False, name='attention')
    no_attention_model = ConvNoAttentionModel(input_size=700, output_size=1, classification=False, name='no_attention')
    two_layer_mlp = BaselineModel(input_size=700, output_size=1, classification=False, name='2_layer_mlp')
    three_layer_mlp = MLPModel(input_size=700, output_size=1, classification=False, name='3_layer_mlp')
    three_layer_mlp_1090 = MLPModel(input_size=701, output_size=1, classification=False, name='3_layer_mlp_1090')
    three_layer_mlp_2 = MLPModel(input_size=700, output_size=1, classification=False, name='3_layer_mlp')

    load_model_weights(three_layer_mlp)
    load_model_weights(three_layer_mlp_2)
    # load_model_weights(three_layer_mlp_1090)

    train(three_layer_mlp_2, X_fft_train, Y_train, epochs=50)
    train(three_layer_mlp, X_train, Y_train, epochs=50)

    linearity_plot(three_layer_mlp_2, data=[X_fft_test, Y_test])
    linearity_plot(three_layer_mlp, data=[X_test, Y_test])

    compare_history([three_layer_mlp, three_layer_mlp_2], metric='val_loss', title='Effect of FFT on ±1500ns time jitter')

    losses = []
    losses.append(tf.keras.losses.MeanAbsoluteError()(Y_test, tf.transpose(three_layer_mlp(X_test, training=False))))
    losses.append(tf.keras.losses.MeanAbsoluteError()(Y_test, tf.transpose(three_layer_mlp_2(X_fft_test, training=False))))
    
    plt.figure(figsize=(10, 6))
    plt.bar(['mlp_no_fft', 'mlp_fft'], losses, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('Mean absolute error')
    plt.title('Model loss from reducing trainable params (on 20k samples after 200 epochs)')
    plt.show()

    return

    jitter_test([three_layer_mlp], X_train, Y_train, epochs=25, plot_jitter=False)

    

    '''
    models = [vgg16, vgg16_small]

    losses = []
    for model in tqdm(models):
        x_test = X_1090_test if '1090' in model.name.split('_') else X_test
        y_test = Y_test
        load_model_weights(model)
        loss = tf.keras.losses.MeanAbsoluteError()(y_test, tf.transpose(model(x_test, training=False)))
        losses.append(loss)

    plt.figure(figsize=(10, 6))
    plt.bar([model.name for model in models], losses, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('Mean absolute error')
    plt.title('Model loss from reducing trainable params (on 20k samples after 200 epochs)')
    plt.show()
    '''

    '''

    models = [two_layer_mlp, three_layer_mlp, three_layer_mlp_1090, two_layer_conv, ten_layer_conv, vgg16_small, vgg13, vgg16, no_attention_model, attention_model]

    for model in models:
        x_train = X_1090_train if '1090' in model.name.split('_') else X_train
        y_train = Y_train
        train(model, x_train, y_train, epochs=100, batch_size=64, callbacks=False, compile=True)
        save_model_weights(model)

    for model in models:
        debug_print(['creating metric plots for', model.name])
        compare_history([model], metric='val_loss', title=model.name + ' model validation loss over training')
        linearity_plot(model, data=[np.squeeze(X_test), Y_test])

    compare_history(models, metric='val_loss', title='Model validation loss over training')
    '''

    # jitter_test([test_model1, test_model2, test_model3, test_model4, baseline_model], X, Y, epochs=100)


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