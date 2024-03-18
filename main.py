import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import sys

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.stats import norm

from preprocessing import *
# from models import *
from regression_models import *
from channel_models import *
from vgg import *
from experiments import *

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
Save and load TF model weights
    model           : Tensorflow model
    name            : Save name

    return          : None
'''
def save_model_weights(model, name=None):
    if not name: name = model.name
    debug_print(['saving', name, 'weights to', MODEL_SAVE_PATH])
    model.save_weights(MODEL_SAVE_PATH + name)


def load_model_weights(model, name=None):
    if not name: name = model.name
    debug_print(['loading', name, 'weights from', MODEL_SAVE_PATH + name + '.data'])
    model.load_weights(MODEL_SAVE_PATH + name)


'''
Generates a linearity plot based on model and data
    model           : Tensorflow model
    data            : (X, Y) tuple
    delta_mu        : Delta mu bin sizes
    num_delta_mu    : Number of delta mu bins
    num_samples     : Number of samples per delta mu bin

    return          : None
'''
def linearity_plot(model, data=None, delta_mu=50, num_delta_mu=20, num_samples=1000, title=''):
    def plot(x, y, err_down, err_up, color='blue', marker='o', title=None):
        plt.plot(x, x, color='black', label='y = x Line')
        plt.errorbar(x, y, yerr=(err_down, err_up), label='\u0394\u03bc Prediction', color=color, marker=marker, linestyle='None')

        plt.xlabel('Actual \u0394\u03bc [ns]')
        plt.ylabel('Median Predicted \u0394\u03bc [ns]')
        plt.title(title)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('figs/' + title)
        plt.clf()
    
    
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
         title=model.name + ' model linearity plot' + title)

'''
Reset learned model weights to initialization
    model           : Tensorflow model

    return          : Reset model
'''
def reset_weights(model):
    debug_print(['resetting', model.name, 'weights'])
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Convolution1D, tf.keras.layers.Dense)):
            layer.set_weights([
                tf.keras.initializers.glorot_normal()(layer.weights[0].shape),
                tf.zeros(layer.weights[1].shape)  # Bias
            ])
    return model

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
    
    X, Y = concat_data(DMS_NAMES)
    # X, Y = shift_distribution(X, Y)
    # plot_distribution(Y)
    data_indices = np.array([i for i in range(len(X))])
    np.random.shuffle(data_indices)
    X = X[data_indices][:num_samples]
    X = np.concatenate([np.expand_dims(normalize(dist), 0) for dist in X], axis=0)
    Y = Y[data_indices][:num_samples]

    # X = jitter_pulses(X, t=10)

    
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
    Models
    
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
    '''

    '''
    Experiments
    '''

    '''
    X_train = jitter_pulses(X_train, t=10)
    X_test = jitter_pulses(X_test, t=10)

    '''

    '''

    X_dev, X_params = get_relative_deviation(X_train)

    latent_size = 512
    autoencoder = AutoEncoder(
        input_size=700, 
        encoder_layer_sizes=[latent_size],
        decoder_layer_sizes=[700]
    )

    # regression_model_large = CustomMLPModel(input_size=700, layer_sizes=[512, 256, 64, 1])
    regression_model_small = CustomMLPModel(input_size=latent_size, layer_sizes=[8, 1])

    train(autoencoder, X_dev, X_dev, epochs=3, batch_size=64)

    for x, params in zip(X_train[:2], X_params[:2]):
        a = np.linspace(0, X.shape[1], X.shape[1])
        x_p = x[:, 0] * gaussian(a, *params) / params[0]
        plt.plot(x, label='Original Pulse')
        plt.plot(x_p, label='Reconstructed Pulse')
        plt.legend()
        plt.show()

    return

    '''

    latent_size = 32
    autoencoder = Autoencoder(
        input_size=700, 
        encoder_layer_sizes=[512, 256, latent_size],
        decoder_layer_sizes=[256, 700]
    )

    '''
    variational_autoencoder = VariationalAutoencoder(
        input_size=700, 
        encoder_layer_sizes=[512, 256, latent_size],
        decoder_layer_sizes=[256, 700]
    )
    for i in range(10):
        variational_autoencoder.train_step(X_train)
    '''

    train(autoencoder, X_train, X_train, epochs=32, batch_size=64)

    for x in X_test[:2]:
        plt.plot(x, label='Original Pulse')
        plt.plot(autoencoder(np.array([x]))[0], label='Reconstructed Pulse')
        plt.legend()
        plt.show()

    encoded_X_train = np.array(autoencoder.encode(X_train), dtype=np.float16)
    debug_print(['Size before compression:', sys.getsizeof(X_train), 'bytes\n',
                 'Size after compression:', sys.getsizeof(encoded_X_train), 'bytes'])

    regression_model_small = CustomMLPModel(input_size=latent_size, layer_sizes=[64, 64, 1])

    train(regression_model_small, encoded_X_train, Y_train, epochs=100, batch_size=64)

    linearity_plot()


    # X_train_compressed = autoencoder.encode_data(X_train)

    # train(regression_model_large, X_train, Y_train, epochs=40, batch_size=64)
    # train(regression_model_small, X_train_compressed, Y_train, epochs=40, batch_size=64)

    # tuner = keras_tuner.RandomSearch(tuner_model, objective='val_loss', max_trials=100)
    # tuner.search(np.squeeze(X_train), Y_train, epochs=35, batch_size=128, validation_data=(np.squeeze(X_test), Y_test))

    # plot_parameter_performance(['untitled_project/', '../untitled_project_2/', '../untitled_project_4/'], title='Number of Parameters vs. Training Performance [val MAE] [3/9/24]')

    '''
    tuner = tfdf.tuner.RandomSearch(num_trials=5, use_predefined_hps=True)
    model = tfdf.keras.RandomForestModel(verbose=2,tuner=tuner)
    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=tf.keras.optimizers.Adam())
    
    train(model, np.squeeze(X_train), Y_train, epochs=1, batch_size=64)
    '''

    # model = CustomMLPModel(input_size=700, layer_sizes=[512, 128, 16, 1], classification=False)
    # train(model, X_fft_train, Y_train, epochs=100, batch_size=64)
    # linearity_plot(model, (X_fft_test, Y_test), num_samples=500, num_delta_mu=30)


    '''
    X_train_jitter = jitter_pulses(X_train, t=300)
    X_test_jitter = jitter_pulses(X_test, 300)
    three_layer_mlp = MLPModel(input_size=700, output_size=1, classification=False, name='3_layer_mlp')
    load_model_weights(three_layer_mlp)

    train(three_layer_mlp, X_train_jitter, Y_train, epochs=100, batch_size=64)
    linearity_plot(three_layer_mlp, (X_test_jitter, Y_test), num_samples=500, num_delta_mu=30)

    three_layer_mlp_2 = MLPModel(input_size=700, output_size=1, classification=False, name='3_layer_mlp')
    load_model_weights(three_layer_mlp)
    # load_model_weights(three_layer_mlp_2)

    X_train_shift, Y_train_shift = shift_distribution(X_train, Y_train)
    print(X_train_shift.shape)

    train(three_layer_mlp_2, X_train_shift, Y_train_shift, epochs=100)

    linearity_plot(three_layer_mlp, (X_test, Y_test))
    linearity_plot(three_layer_mlp_2, (X_test, Y_test))
    '''
    
        
    # mlp_jitter_test(X_train, Y_train, X_test, Y_test, epochs=100)

    '''
    three_layer_mlp = MLPModel(input_size=700, output_size=1, classification=False, name='3_layer_mlp')
    load_model_weights(three_layer_mlp)

    inputs = X[0] - X_dist
    compute_saliency_map(three_layer_mlp, inputs, baseline=X_dist, title='Saliency map of example pulse')

    '''

    '''

    model = CustomMLPModel(700, [512, 256, 64, 1])
    all_losses = []
    final_losses = []
    data_sizes = [5000, 10000]
    for data_size in data_sizes:
        sz = int(data_size * 0.8)
        history = train(model, X_train[:sz], Y_train[:sz], epochs=10, batch_size=128)
        losses = history.history['val_loss']
        final_losses.append(losses[-1])
        all_losses.append(losses)
        linearity_plot(model, (X_test, Y_test), title=' ' + str(data_size) + ' ')
        reset_weights(model)

    for data_size, losses in zip(data_sizes, all_losses):
        plt.plot(losses, label='data size ' + str(data_size))
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.title(model.name + ' Training Convergence vs. Data Size')
    plt.show()
    
    plt.plot(final_losses)
    plt.xlabel('Data Size')
    plt.ylabel('Loss (MAE)')
    plt.title(model.name + ' Training Loss vs. Data Size')
    plt.show()

    '''

    # train(model, X_train, Y_train, epochs=100, batch_size=64)
    # linearity_plot(model, (X_test, Y_test), title=model.name + ' linearity test [new data]')
        
    
    '''
    convert_files_to_gif('gif/', 'saliency_map.gif')
    '''

    '''

    '''

    # mlp_jitter_test(X_train, Y_train, X_test, Y_test, epochs=100)

    # linearity_plot(three_layer_mlp, data=(X_test, Y_test))

    # train(three_layer_mlp, X_train, Y_train, epochs=50, batch_size=32, compile=True, metrics=[tf.keras.losses.MeanAbsoluteError()])

    # linearity_plot(three_layer_mlp, data=(X_test, Y_test))

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