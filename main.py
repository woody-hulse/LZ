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

DATE = ' [4-19-24]'

# import importlib
# importlib.reload(regression_models)

DSS_NAME = '../dSSdMS/dSS_230918_gaussgas_700sample_area7000_1e5events_random_centered.npz'
DMS_NAME = '../dSSdMS/dMS_231202_gaussgas_700sample_area7000_areafrac0o5_deltamuinterval50ns_5000each_5e4events_random_centered_above1000ns_batch00.npz'
DMS_NAMES = [
    '/Users/woodyhulse/Documents/lz/dSSdMS/dMS_240306_gaussgas_700sample_148electrons_randomareafrac_deltamuinterval50ns_5000each_1e5events_random_jitter100ns_batch00.npz',
    # '/Users/woodyhulse/Documents/lz/dSSdMS/dMS_240306_gaussgas_700sample_148electrons_randomareafrac_deltamuinterval50ns_5000each_1e5events_random_jitter100ns_batch01.npz',
    # '/Users/woodyhulse/Documents/lz/dSSdMS/dMS_240306_gaussgas_700sample_148electrons_randomareafrac_deltamuinterval50ns_5000each_1e5events_random_jitter100ns_batch02.npz'
]
DMS_NAMES = ['../dSSdMS/dMS_231011_gaussgas_700sample_area7000_areafrac0o5_deltamuinterval50ns_5000each_1e5events_random_centered_batch10.npz']
DSS_SIMPLE_NAME = '/Users/woodyhulse/Documents/lz/dSSdMS/dSS_2400419_gaussgass_700samplearea7000_areafrac0o5_1.0e+05events_random_centered_withEAT_1e.npz'
DMS_AT_NAME = '../dSSdMS/dSS_2400417_gaussgass_700samplearea7000_areafrac0o5_1.0e+05events_random_centered_channel_withEAT_10e.npz'
DMS_AT_CHANNEL_NAME = '../dSSdMS/dSS_2400417_gaussgass_700samplearea7000_areafrac0o5_1.0e+05events_random_centered_channel_withEAT.npz'
DMS_CHANNEL_NAME = '../dSSdMS/dSS_2400417_gaussgass_700samplearea7000_areafrac0o5_1.0e+05events_random_centered_channel_withEAT.npz'

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
    
    return history

'''
Perform classification-based tasks and experiments
'''
def classification():
    num_samples = 30000

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
    num_samples = 50000

    def normalize(data):
        if np.linalg.norm(data) == 0:
            print('ALERT', data)
            return None
        else: return data / np.linalg.norm(data)

    debug_print(['preprocessing data'])

    X = np.array([None for _ in range(int(1e6))])
    Y = np.array([None for _ in range(int(1e6))])
    areafrac = np.array([None for _ in range(int(1e6))])
    AT = np.array([None for _ in range(int(1e6))])
    
    # X, Y, AT = generate_ms_pulse_dataset(num_samples, arrival_times=True, save=True)
    # X, Y, AT = generate_ms_pulse_dataset_multiproc(num_samples, arrival_times=True, save=True)
    X, XC, Y, AT = load_pulse_dataset(DSS_SIMPLE_NAME)
    # XS = np.sum(XC, axis=(1, 2))
    # XB = np.max(XC, axis=(1, 2))

    '''
    image_frames = []
    imgs = np.transpose(XC[:10], axes=[0, 3, 1, 2])
    for t in imgs[0]:
        plt.imshow(t)
        plt.title('Hit pattern')
        
        plt.gcf().canvas.draw()
        width, height = plt.gcf().get_size_inches() * plt.gcf().dpi
        data = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
        image_array = data.reshape(int(height), int(width), 3)

        image_frame = Image.fromarray(image_array)
        image_frames.append(image_frame)

    image_frames[0].save('hit_pattern' + '.gif', 
                        save_all = True, 
                        duration = 20,
                        loop = 0,
                        append_images = image_frames[1:])
    '''
    

    # X, Y, _ = concat_data(DMS_NAMES)
    # X, Y = shift_distribution(X, Y)
    # plot_distribution(Y)
    data_indices = np.array([i for i in range(min(X.shape[0], num_samples))])
    np.random.shuffle(data_indices)
    X = X[data_indices][:num_samples]
    # XC = XC[data_indices][:num_samples]
    Y = Y[data_indices][:num_samples]
    AT = AT[data_indices][:num_samples]
    areafrac = areafrac[data_indices][:num_samples]

    # AT_hist = at_to_hist(AT)

    '''
    X2, Y2 = generate_ms_pulse_dataset(num_samples)

    di = np.array([i for i in range(len(X2))])
    np.random.shuffle(di)
    X2 = X2[di]
    X2 = np.concatenate([np.expand_dims(normalize(dist), 0) for dist in X2], axis=0)

    for i1 in range(25):
        x1, y1 = X[i1], Y[i1]

        y2 = -1
        while y2 != y1:
            i2 = np.random.randint(0, num_samples)
            x2, y2 = X2[i2], Y2[i2]
            print(y1, y2)
        
        print(y1)
        plt.plot(x1)
        plt.plot(x2)
        plt.show()
    '''

    # X = jitter_pulses(X, t=20)
    '''
    X_avg = normalize(np.average(X, axis=0))
    x = np.linspace(0, X.shape[1], X.shape[1])
    params, cov = curve_fit(gaussian, x, X_avg)
    X_dist = gaussian(x, *params)
    X_dist = np.expand_dims(X_dist, axis=-1)
    '''
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

    # X = np.expand_dims(X, axis=-1)
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # X_fft_train, X_fft_test, Y_train, Y_test = train_test_split(X_fft, Y, test_size=0.2, random_state=42)
    # X_1090_train = add_1090(X_train)
    # X_1090_test = add_1090(X_test)

    # debug_print(['     X_train:', X_train.shape])
    # debug_print(['     Y_train:', Y_train.shape])

    '''
    Experiments
    '''


    '''
    model = CustomMLPModel(700, layer_sizes=[700, 700, 700])
    train(model, XS, X, epochs=15, batch_size=512, callbacks=True)

    for x, xb in zip(X[:10], XS[:10]):
        x_hat = model(np.expand_dims(xb, axis=0))[0]
        plt.plot(x_hat, label='Reconstructed pulse')
        plt.plot(x, label='True pulse')
        plt.legend()
        plt.show()
        plt.plot(xb)
        plt.title('Binary pulse' + DATE)
        plt.show()
    '''


    '''
    ATH = np.array([[0]] for at in AT[:10])
    ATH = np.zeros(X.shape)
    for i, at in tqdm(enumerate(AT)):
        hist, bins = np.histogram(at, bins=np.arange(701))
        ATH[i] = hist

    at_model = CustomMLPBinnedModel(700, layer_sizes=[700, 700, 700])
    train(at_model, X, ATH, epochs=50, batch_size=128, callbacks=True)

    for i in range(10):
        at = ATH[i]
        at_hat = at_model(np.expand_dims(X[i], axis=0))[0]
        plot_at_hists(at_hat, label='Predicted arrival times')
        plot_at_hists(at, label='True arrival times')
        plt.title('True vs predicted histogram of electron arrival times')
        plt.xlabel('Arrival time (samples, 10ns)')
        plt.ylabel('Number of electrons')
        plt.legend()
        plt.show()
    '''

    '''
    # channel_model =  MLPChannelModel(input_size=700, head_sizes=[256, 128, 16], layer_sizes=[256, 256, 1], heads=5)
    channel_model = ConvChannelModel(input_size=list(XC.shape[1:]))
    train(channel_model, XC, Y, epochs=100, batch_size=128, callbacks=True, summary=True)

    # linearity_plot(channel_model, (XC, Y))

    model = CustomMLPModel(700, layer_sizes=[512, 256, 128, 1])
    train(model, X, Y, epochs=100, batch_size=128)
    '''

    '''
    model = CustomMLPModel(700, layer_sizes=[512, 256, 256, NUM_ELECTRONS])
    train(model, X, AT, epochs=10, batch_size=128)

    test_samples = 10
    for x, dmu, at in zip(X[:test_samples], Y[:test_samples], AT[:test_samples]):
        at_hat = model(np.expand_dims(x, axis=0))[0]
        plt.title(f'True vs predicted electron arrival times for Δμ={dmu} pulse')
        plt.plot(x, label='Event')
        plt.axvline(x=at_hat, color='#CC4F1B',  linestyle='--', label='Arrival time prediction')
        # plt.plot(at, marker='+', label='True electron arrival times')
        # plt.plot(at_hat, marker='o', label='Predicted electron arrival times')
        # plt.ylabel('Arrival time (samples, 10ns)')
        plt.xlabel('Sample')
        # plt.ylim((0, 700))
        plt.legend()
        plt.show()

        continue 
    '''

    electron_counts = [256]
    errors = []
    for n in electron_counts:
        X, XC, Y, AT = generate_pulse_dataset_multiproc(100000, bins=20, max_delta_mu=0, arrival_times=True, save=False, task=pulse_task, num_electrons=n)

        model = CustomMLPModel(700, layer_sizes=[512, 256, 256, n])
        train(model, X, AT, epochs=128, batch_size=256, summary=True)

        x = X[:500]
        at = AT[:500]
        at_ = model(X[:500])
        error = tf.keras.losses.MeanSquaredError()(at, at_).numpy()
        error_scatterplot(at, at_)
        errors.append(error)

        test_samples=4
        for i, (x, dmu, at) in enumerate(zip(X[:test_samples], Y[:test_samples], AT[:test_samples])):
            at_hat = model(np.expand_dims(x, axis=0))[0]
            plt.title(f'True vs predicted electron arrival times for Δμ={dmu} pulse')
            plt.plot(at, marker='+', label='True electron arrival times')
            plt.plot(at_hat, marker='o', label='Predicted electron arrival times')
            plt.ylabel('Arrival time (samples, 10ns)')
            plt.xlabel('Sample')
            # plt.ylim((0, 700))
            plt.legend()
            plt.savefig(f'eat_{n}_{i}')
            plt.clf()
    
    fig, ax = plt.subplots()
    ax.plot(electron_counts, errors, marker='+')
    ax.set_ylabel('MSE')
    ax.set_xlabel('Event electron count')
    ax.set_xscale('log', base=2)
    ax.set_title('Number of electrons in event vs arrival time prediction accuracy')
    plt.show()

    '''
    at_model = CustomMLPModel(700, layer_sizes=[512, 256, 256, NUM_ELECTRONS])
    # at_model = ConvChannelModel(700, layer_sizes=[1024, 512, 256, NUM_ELECTRONS])
    # at_model = MLPChannelModel(input_size=700, head_sizes=[256, 128, 16], layer_sizes=[256, 256, NUM_ELECTRONS], heads=5)
    train(at_model, X, AT, epochs=300, batch_size=256, callbacks=False, summary=True)
    # AT_hat = at_model(X[:10000])
    # residual_plot(AT[:10000], AT_hat)
    # error_scatterplot(AT[:10000], AT_hat)

    
    for i in range(10):
        at = AT[i]
        at_hat = at_model(np.expand_dims(X[i], axis=0))[0]

        plt.title('True vs predicted electron arrival times for Δμ=0 pulse')
        plt.plot(at, label='True electron arrival times')
        plt.plot(at_hat, label='Predicted electron arrival times')
        plt.ylabel('Arrival time (samples, 10ns)')
        plt.xlabel('Electron number')
        plt.legend()
        plt.show()
    '''

    '''
        at_hat_hist, _ = np.histogram(at_hat, bins=np.arange(0, 700, 1))
        at_hist, _ = np.histogram(at, bins=np.arange(0, 700, 1))
        plot_at_hists(at_hat_hist, label='Predicted arrival times')
        plot_at_hists(at_hist, label='True arrival times')
        plt.title('True vs predicted histogram of electron arrival times')
        plt.xlabel('Arrival time (samples, 10ns)')
        plt.ylabel('Number of electrons')
        plt.legend()
        plt.show()
        '''

    '''

    test_samples = 10
    for x, dmu, at in zip(X[:test_samples], Y[:test_samples], AT[:test_samples]):
        at_hat = at_model(np.expand_dims(x, axis=0))[0]
        plt.title(f'True vs predicted electron arrival times for Δμ={dmu} pulse')
        plt.plot(at, marker='+', label='True electron arrival times')
        plt.plot(at_hat, marker='o', label='Predicted electron arrival times')
        plt.ylabel('Arrival time (samples, 10ns)')
        plt.xlabel('Electron number')
        plt.legend()
        plt.show()

        continue

    at_model = CustomMLPModel(700, layer_sizes=[2048, 1024, 256, NUM_ELECTRONS])



    at_model = ConvChannelModel(700, layer_sizes=[1024, 512, 256, NUM_ELECTRONS])
    train(at_model, XC, AT, epochs=100, batch_size=128)

    test_samples = 10
    for x, dmu, at in zip(XC[:test_samples], Y[:test_samples], AT[:test_samples]):
        at_hat = at_model(np.expand_dims(x, axis=0))[0]
        plt.title(f'True vs predicted electron arrival times for Δμ={dmu} pulse')
        plt.plot(at, marker='+', label='True electron arrival times')
        plt.plot(at_hat, marker='o', label='Predicted electron arrival times')
        plt.ylabel('Arrival time (samples, 10ns)')
        plt.xlabel('Electron number')
        plt.legend()
        plt.show()

        continue
    '''
        
    

    '''

    ae = Autoencoder(
        input_size=700, 
        encoder_layer_sizes=[512, 256, 32],
        decoder_layer_sizes=[256, 700]
    )

    vqvae = VQVariationalAutoencoder(
        input_size=700, 
        num_embeddings=65536, 
        encoder_layer_sizes=[512, 256, 128],
        decoder_layer_sizes=[256, 700]
    )

    # train(ae, X_train, X_train, epochs=20, batch_size=128)
    train(vqvae, X_train, X_train, epochs=20, batch_size=256)

    for x in X[:4]:
        x_hat = vqvae(np.array([x]))[0]
        plt.plot(x)
        plt.plot(x_hat)
        plt.show()

    '''

    '''
    
    at_weight = 1e-7
    mhae = MultiHeaddedAutoencoder(
        input_size=700, 
        encoder_layer_sizes=[512, 256, 16], 
        decoders=[[256, 512, 700], [128, 128, 148]],
        loss_weights=[1 - at_weight, at_weight])
    
    
    
    # train(mhae, X, [X, AT], epochs=20, batch_size=128, summary=True)

    compare_latent_dim_compression_at([1, 2, 4, 8, 16, 32, 64, 128, 256], X, Y, AT, X[:100], Y[:100], AT[:100])

    at_model = CustomMLPModel(input_size=np.prod(X.shape[1:]), layer_sizes=[512, 256, 256, NUM_ELECTRONS])
    train(at_model, X, AT, epochs=10, batch_size=4, summary=True)

    test_samples = 10
    for x, dmu, at in zip(X[:test_samples], Y[:test_samples], AT[:test_samples]):
        at_hat = at_model(np.expand_dims(x, axis=0))[0]
        plt.title(f'True vs predicted electron arrival times for Δμ={dmu} pulse')
        plt.plot(at, marker='+', label='True electron arrival times')
        plt.plot(at_hat, marker='o', label='Predicted electron arrival times')
        plt.ylabel('Arrival time (samples, 10ns)')
        plt.xlabel('Electron number')
        plt.legend()
        plt.show()

        continue
        
        at_hist, _ = np.histogram(at, bins=np.arange(0, 700, 1))
        at_hat_hist, _ = np.histogram(at_hat, bins=np.arange(0, 700, 1))
        plot_at_hists(at_hist, label='True arrival times')
        plot_at_hists(at_hat_hist, label='Predicted arrival times')
        plt.title('True vs predicted histogram of electron arrival times')
        plt.xlabel('Arrival time (samples, 10ns)')
        plt.ylabel('Number of electrons')
        plt.legend()
        plt.show()

    '''

    # compare_latent_dim_compression([1, 2, 4, 8, 16, 32, 64, 128, 256], X_train, Y_train, X_test, Y_test)


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