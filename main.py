import os
import datetime

import pandas as pd
import numpy as np

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

DSS_NAME = 'dSS_230918_gaussgas_700sample_area7000_1e5events_random_centered'
DMS_NAME = 'dMS_230918_gaussgas_700sample_area7000_areafrac0o3_deltamuall50ns_5000each_1e5events_random_centered'


class DS():
    def __init__(self, DSname):
        self.DSname = DSname
        self.DStype = 'test'

        path ='../dSSdMS/' + self.DSname + '.npz'
        with np.load(path) as f:
            debug_print(['loading', self.DStype, 'data from', path])
            fkeys = f.files
            self.DSdata = f[fkeys[0]]       # arraysize =（event * 700 samples)
            self.ULvalues = f[fkeys[1]]    # arraysize =（event * 4 underlying parameters)

            # print(self.DSdata.shape, self.DSdata)
            # print(self.ULvalues.shape, self.ULvalues)


def debug_print(statements=[], end='\n'):
    ct = datetime.datetime.now()
    print('[', str(ct)[:19], '] ', sep='', end='')
    for statement in statements:
        print(statement, end=' ')
    print(end=end)


def gaussian(X, C, sigma,):
    return C*np.exp(-(X-350)**2/(2*sigma**2)) # hard-coded mean


def jitter_pulses(pulses, t=50):
    debug_print(['jittering pulses'])
    jitter_pulses = []
    for pulse in pulses:
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


def train(model, X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, compile=False, summary=True, callbacks=True, learning_rate=0.001, metrics=[]):
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
        c.append(tf.keras.callbacks.EarlyStopping(patience=10))
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

    def normalize(data):
        if np.linalg.norm(data) == 0:
            print('ALERT', data)
            return None
        else: return data / np.linalg.norm(data)

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

    # train(baseline_model, X_train, Y_train, epochs=50, batch_size=16, callbacks=False)
    train(test_model, X_train, Y_train, epochs=50, batch_size=128, callbacks=False)
    # train(test_model2, X_dev_train, Y_train, epochs=25, batch_size=16, compile=True)
    # train(test_model3, X_diff_train, Y_train, epochs=25, batch_size=16, compile=True)


def jitter_test(models, X, Y, epochs=100):

    debug_print(['running jitter test'])

    sns.set_theme(style="whitegrid")

    jitter = [0, 5, 20, 100, 200]
    X_jitter = []
    for j in jitter:
        X_jitter.append(jitter_pulses(X, j))

    losses = []
    for index, model in enumerate(models):
        for i, (x, j) in enumerate(zip(X_jitter, jitter)):
            debug_print(['model', index + 1, '/', len(models), ':', model.name, '- test', i + 1, 'of', len(jitter)])
            model = reset_weights(model)
            history = train(model, x, Y, epochs=epochs, batch_size=16, validation_split=0.2, compile=False, summary=False)
            loss = history.history['val_loss']
            losses.append([j, model.name, round(loss[-1], 3)])
    df = pd.DataFrame(losses, columns=['Jitter', 'Model', 'MAE Loss: ' + str(epochs) + ' epochs'])

    palette = sns.color_palette("rocket_r")
    g = sns.catplot(data=df, kind='bar', x='Model', y='MAE Loss: ' + str(epochs) + ' epochs', hue='Jitter', errorbar='sd', palette=palette, alpha=0.7, height=6)
    g.fig.suptitle('Model sensitivity to temporal variation')
    g.despine(left=True)
    g.set_axis_labels('', 'Loss (MAE) after ' + str(epochs) + ' epochs')
    g.legend.set_title('Variation (\u0394t)')
    plt.show()
    

def regression():
    num_samples = 1000

    dMS = DS(DMS_NAME)

    def normalize(data):
        if np.linalg.norm(data) == 0:
            print('ALERT', data)
            return None
        else: return data / np.linalg.norm(data)

    debug_print(['preprocessing data'])
    data_indices = np.array([i for i in range(len(dMS.ULvalues))])
    np.random.shuffle(data_indices)
    X = dMS.DSdata[data_indices][:num_samples]
    X = np.concatenate([np.expand_dims(normalize(dist), 0) for dist in X], axis=0)

    '''
    X_avg = normalize(np.average(X, axis=0))
    x = np.linspace(0, X.shape[1], X.shape[1])
    params, cov = curve_fit(gaussian, x, X_avg)
    X_dist = gaussian(x, *params)
    X_dev = np.concatenate([np.expand_dims(X[i] / X_dist, axis=0) for i in range(X.shape[0])], axis=0)
    X_diff = np.concatenate([np.expand_dims(X[i] - X_dist, axis=0) for i in range(X.shape[0])], axis=0)
    '''

    X = np.expand_dims(X, axis=-1)
    Y = np.array(dMS.ULvalues)[:, 1][data_indices][:num_samples]

    '''
    X_dev_train, X_dev_test, Y_train, Y_test = train_test_split(X_dev, Y, test_size=0.2, random_state=42)
    X_diff_train, X_diff_test, Y_train, Y_test = train_test_split(X_diff, Y, test_size=0.2, random_state=42)
    '''

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    debug_print(['     X_train:', X_train.shape])
    debug_print(['     Y_train:', Y_train.shape])

    test_model1 = ConvModel(input_size=700)
    test_model2 = BaselineConvModel(output_size=1).build_model()
    # test_model3 = MLPModel()
    baseline_model = BaselineModel(output_size=1).build_model() 
    test_model3 = build_vgg(length=700, name='vgg13', width=16)
    test_model4 = build_vgg(length=700, name='vgg16', width=16)

    jitter_test([test_model1, test_model2, test_model3, test_model4, baseline_model], X, Y, epochs=100)

    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    # my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=100), reduce_lr]
    # training_history = baseline_model.fit(X_train, Y_train, epochs=100, validation_split=0.2, callbacks=[my_callbacks])

    # train(baseline_model, X_train, Y_train, epochs=100, batch_size=16, compile=True)
    # train(test_model3, X_train, Y_train, epochs=50, batch_size=16)
    # train(baseline_model, X_train, Y_train, epochs=50, batch_size=16)
    # train(test_model2, X_dev_train, Y_train, epochs=25, batch_size=16, compile=True)
    # train(test_model3, X_diff_train, Y_train, epochs=25, batch_size=16, compile=True)


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
    classification()


if __name__ == '__main__':
    # os.system('clear')
    main()