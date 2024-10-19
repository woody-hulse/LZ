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

from main import *
sns.set_style(style='whitegrid',rc={'font.family': 'sans-serif','font.serif':'Times'})
sns.color_palette('hls', 8)

'''
Plot scatterplot of errors
'''
def error_scatterplot(y_true, y_pred, num_scatter=500):
    error = np.square(y_true - y_pred)
    sign = (y_true - y_pred) / np.abs(y_true - y_pred)
    signed_error = error * sign
    mean_signed_error = np.mean(signed_error, axis=0)

    x = np.arange(y_true.shape[1])
    for errors in signed_error[:num_scatter]:
        plt.scatter(x, errors, alpha=0.2, color='#FF9848', s=5)

    plt.scatter(x, mean_signed_error, color='#FF9848', s=10)

    plt.plot(x, mean_signed_error, color='#CC4F1B', label='Mean signed error')
    plt.title('Signed mean squared error of arrival times' + DATE)
    plt.ylim((-1000, 1000))
    plt.ylabel('Signed squared error')
    plt.xlabel('Electron index')
    plt.legend()
    # plt.show()
    plt.savefig(f'figs/smse_p_noprotondiff_{y_true.shape[1]}')
    plt.clf()


'''
Plot average residuals
'''
def residual_plot(y_true, y_pred):
    diff = y_true - y_pred
    residual = np.square(diff)
    avg_residual = np.mean(residual, axis=0)
    med_diff = np.median(diff, axis=0)
    avg_diff = np.mean(diff)
    std_residual = np.std(residual, axis=0)

    print('mean:', np.mean(y_true - y_pred))

    diff_16 = np.zeros(y_true.shape[1])
    diff_84 = np.zeros(y_true.shape[1])
    for i in range(y_true.shape[1]):
        hist, bin_edges = np.histogram(diff[:, i], bins=100, density=True)
        cumulative_distribution = np.cumsum(hist * np.diff(bin_edges))
        total_area = cumulative_distribution[-1]
        diff_16[i] = bin_edges[np.where(cumulative_distribution >= 0.16 * total_area)[0][0]]
        diff_84[i] = bin_edges[np.where(cumulative_distribution >= 0.84 * total_area)[0][0]]

    plt.title('Electron arrival time prediction' + DATE)
    x = np.arange(y_true.shape[1])
    y = np.zeros_like(x)
    plt.plot(med_diff, color='#CC4F1B', label='Difference between real and predicted value')
    # plt.fill_between(y, med_diff, color='blue', label='Difference between real and predicted value')
    # plt.errorbar(x, med_diff, yerr=(diff_16, diff_84), label='Error of predicted value', color='black', linestyle='None')
    plt.plot(y, linestyle='dashed', color='black', label='True')
    plt.fill_between(x, diff_16, diff_84, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='16/84 error in prediction')
    plt.ylim((-10, 10))
    plt.legend()
    plt.show()




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
    X_delta_mu = np.empty((num_delta_mu, num_samples))

    if data:
        X_delta_mu = np.empty([num_delta_mu, num_samples] + list(X.shape[1:]))

    for i in tqdm(range(len(X))):
        index = int(Y[i] // delta_mu)
        if counts[index] >= num_samples: continue
        else:
            X_delta_mu[index, int(counts[index])] = X[i]
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
Plot histogram for binned events
'''
def plot_histogram(X_binned, Y_binned, model):
    num_bins = X_binned.shape[0]
    assert num_bins == 20

    fig, ax = plt.subplots(5, 4, figsize=(12, 12))

    for i in tqdm(range(num_bins)):
        row = i // 4
        col = i % 4
        X = X_binned[i]
        y = Y_binned[i][0]
        Y_hat = model(X)[:, 0]

        ax[row][col].hist(Y_hat, bins=100, color='lightblue', edgecolor='lightblue', label='Predicted delta mu')
        ax[row][col].axvline(y, color='red', label='True delta mu', linestyle='dashed')
        ax[row][col].set_xlim(-10, 1010)
        ax[row][col].set_title(f'Δμ = {y}ns')
    
    ax[0][0].set_xlabel('Predicted delta mu')
    ax[0][0].set_ylabel('Count')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f'Delta mu predictions for each godnumber delta mu', fontsize=13)
    plt.show()

'''
Plot confidence histogram for binned events
'''
def plot_confidence_histogram(X_binned, Y_binned, model):
    num_bins = X_binned.shape[0]
    assert num_bins == 20

    fig, ax = plt.subplots(5, 4, figsize=(12, 12))

    for i in tqdm(range(num_bins)):
        row = i // 4
        col = i % 4
        X = X_binned[i]
        y = Y_binned[i][0]
        Y_output = model(X)
        Y_hat, Y_hat_confidence = Y_output[:, 0], Y_output[:, 1]

        Y_hat_confident = Y_hat[Y_hat_confidence > 0.5]

        ax[row][col].hist(Y_hat, bins=100, color='lightblue', edgecolor='lightblue', label='Predicted delta mu (confidence < 0.5)')
        ax[row][col].hist(Y_hat_confident, bins=100, color='green', edgecolor='green', label='Predicted delta mu (confidence ≥ 0.5)')
        ax[row][col].axvline(y, color='red', label='True delta mu', linestyle='dashed')
        ax[row][col].set_xlim(-10, 1010)
        ax[row][col].set_title(f'Δμ = {y}ns')
    
    ax[0][0].set_xlabel('Predicted delta mu')
    ax[0][0].set_ylabel('Count')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f'Delta mu predictions for each godnumber delta mu', fontsize=13)
    plt.legend()
    plt.show()

'''
Plot confidence histogram for binned events
'''
def plot_confidence_gradient_histogram(X_binned, Y_binned, model):
    num_bins = X_binned.shape[0]
    assert num_bins == 20

    fig, ax = plt.subplots(5, 4, figsize=(12, 12))

    for i in tqdm(range(num_bins)):
        row = i // 4
        col = i % 4
        X = X_binned[i]
        y = Y_binned[i][0]
        Y_output = model(X)
        Y_hat, Y_hat_confidence = Y_output[:, 0], Y_output[:, 1]

        interval = 10
        for i in range(0, interval, 1):
            conf_thresh = i / interval
            Y_hat_confident = Y_hat[Y_hat_confidence >= conf_thresh]
            color = cm.viridis(conf_thresh)
            ax[row][col].hist(Y_hat_confident, bins=100, color=color, edgecolor=color, label=f'Predicted delta mu (confidence ≥ {conf_thresh})')

        ax[row][col].axvline(y, color='red', label='True delta mu', linestyle='dashed')
        ax[row][col].set_xlim(-10, 1010)
        ax[row][col].set_title(f'Δμ = {y}ns')
    
    ax[0][0].set_xlabel('Predicted delta mu')
    ax[0][0].set_ylabel('Count')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f'Delta mu predictions for each godnumber delta mu', fontsize=13)
    plt.show()


def plot_pdf_gradient_histogram(X_binned, Y_binned, model):
    num_bins = X_binned.shape[0]
    assert num_bins == 20

    fig, ax = plt.subplots(5, 4, figsize=(12, 12))

    for i in tqdm(range(num_bins)):
        row = i // 4
        col = i % 4
        X = X_binned[i]
        y = Y_binned[i][0]
        Y_output = model(X)
        Y_hat_mu, Y_hat_sigma = Y_output[:, 0] * 1000, Y_output[:, 1] * 1000

        interval = 20
        for i in range(interval):
            conf_thresh = i / interval
            Y_hat_confident = Y_hat_mu[Y_hat_sigma / 100 >= conf_thresh]
            color = cm.viridis(conf_thresh)
            ax[row][col].hist(Y_hat_confident, bins=100, color=color, edgecolor=color, label=f'Predicted delta mu ≥ {conf_thresh})')

        ax[row][col].axvline(y, color='red', label='True delta mu', linestyle='dashed')
        ax[row][col].set_xlim(-10, 1010)
        ax[row][col].set_ylim(-0.1, 250)
        ax[row][col].set_title(f'Δμ = {y}ns')
    
    ax[0][0].set_xlabel('Predicted delta mu')
    ax[0][0].set_ylabel('PDF')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f'Delta mu predictions for each godnumber delta mu', fontsize=13)
    plt.show()


def plot_distribution_histogram(X_binned, Y_binned, model, pdf_func):
    num_bins = X_binned.shape[0]
    assert num_bins == 20

    fig, ax = plt.subplots(5, 4, figsize=(12, 12))

    for i in tqdm(range(num_bins)):
        row = i // 4
        col = i % 4
        X = X_binned[i]
        y = Y_binned[i][0]
        Y_output = model(X)
        
        for y_output in Y_output[:200]:
            arange = np.arange(0, 1000, 1)
            y_hat = pdf_func(arange / 1000, np.expand_dims(y_output, axis=0))
            ax[row][col].fill_between(arange, y_hat, color='blue', alpha=0.01)
        
        ax[row][col].axvline(y, color='red', label='True delta mu', linestyle='dashed')
        ax[row][col].set_xlim(-10, 1010)
        ax[row][col].set_ylim(-0.1, 12)
        ax[row][col].set_title(f'Δμ = {y}ns')
    
    ax[0][0].set_xlabel('Predicted delta mu')
    ax[0][0].set_ylabel('Count')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f'Delta mu predictions for each godnumber delta mu', fontsize=13)
    plt.show()



def plot_distribution_model_examples(X, Y, model, pdf_func, num_examples=10):
    for sample, dmu in zip(X[:num_examples], Y[:num_examples]):
        output = model(np.expand_dims(sample, axis=0))
        y_hat_mu = output[0, 0] * 1000
        x = np.arange(0, 1000, 1)
        y = pdf_func(x / 1000, output)

        plt.tight_layout()
        plt.title(f'Predicted delta mu: {y_hat_mu:.2f}ns, True delta mu: {dmu:.2f}ns')
        plt.plot(x, y, color='blue', label='Δμ distribution')
        plt.axvline(dmu, label=f'True Δμ: {dmu}ns', linestyle='dashed', color='red')
        plt.fill_between(x, y, color='blue', alpha=0.5)
        plt.xlabel('Δμ [ns]')
        plt.ylabel('PDF')
        plt.xlim(-5, 1005)
        plt.ylim(0, 10)
        plt.legend()
        plt.show()


def plot_prediction_z_distribution_normal(X, Y, model, pdf_func):
    Y_hat = np.zeros((Y.shape[0], model.output_size))
    for i in tqdm(range(0, X.shape[0], 512)):
        Y_hat[i:i+512] = model(X[i:i+512])
    
    Y_hat_mu = Y_hat[:, 0] * 1000
    Y_hat_sigma = Y_hat[:, 1] * 1000

    Y_z = (Y - Y_hat_mu) / Y_hat_sigma
    Y_z = Y_z[np.abs(Y_z) < 5]

    plt.figure(figsize=(8, 6))
    plt.tight_layout()
    plt.title('Z scores of delta mu in predicted normal distributions')
    plt.hist(Y_z, bins=100, color='blue', edgecolor='blue', label='z-scores')
    plt.plot(np.arange(-5, 5, 0.1), pdf_func(np.arange(-5, 5, 0.1), np.array([[0., 1., 0.]])) * Y.shape[0] * 0.095, color='red', label='Target distribution')
    plt.xlabel('z-score')
    plt.ylabel('Count')
    plt.xlim(-4, 4)
    plt.legend()
    plt.show()


def plot_prediction_percentile_distribution_normal(X, Y, model, pdf_func):
    Y_hat = np.zeros((Y.shape[0], model.output_size))
    for i in tqdm(range(0, X.shape[0], 512)):
        Y_hat[i:i+512] = model(X[i:i+512])
    
    Y_hat_pdfs = np.zeros((Y.shape[0], 1000))
    for i in tqdm(range(Y.shape[0])):
        Y_hat_pdfs[i] = pdf_func(np.arange(0, 1, 0.001), np.expand_dims(Y_hat[i], axis=0))
    Y_hat_cdfs = np.cumsum(Y_hat_pdfs, axis=1)
    Y_percentiles = np.array([Y_hat_cdfs[i][int(mu)] for i, mu in enumerate(Y)]) / 1000
    
    bins = 100
    plt.figure(figsize=(8, 6))
    plt.tight_layout()
    plt.title('Percentiles of delta mu in predicted nonnegative normal distributions')
    plt.hist(Y_percentiles, bins=bins, color='blue', edgecolor='blue', label='Percentiles')
    plt.axhline(Y.shape[0] / bins, color='red', label='Target distribution')
    plt.xlabel('Percentile')
    plt.ylabel('Count')
    plt.xlim(0, 1)
    plt.legend()
    plt.show()
    



def plot_prediction_z_distribution_skewnormal(X, Y, model, pdf_func):
    Y_hat = np.zeros((Y.shape[0], model.output_size))
    for i in tqdm(range(0, X.shape[0], 512)):
        Y_hat[i:i+512] = model(X[i:i+512])
    
    Y_hat_loc = Y_hat[:, 0] * 1000
    Y_hat_scale = Y_hat[:, 1] * 1000
    Y_hat_alpha = Y_hat[:, 2]

    Y_hat_mu = Y_hat_loc + Y_hat_scale * Y_hat_alpha * np.sqrt(2 / np.pi)
    Y_hat_sigma = np.sqrt(Y_hat_scale ** 2 * (1 - 2 * Y_hat_alpha ** 2 / np.pi))

    Y_z = (Y - Y_hat_mu) / Y_hat_sigma
    Y_z = Y_z[np.abs(Y_z) < 5]

    plt.figure(figsize=(8, 6))
    plt.tight_layout()
    plt.title('Z scores of delta mu in predicted skew normal distributions')
    plt.hist(Y_z, bins=100, color='blue', edgecolor='blue', label='z-scores')
    plt.plot(np.arange(-5, 5, 0.1), pdf_func(np.arange(-5, 5, 0.1), np.array([[0., 1., 0.]])) * Y.shape[0] * 0.095, color='red', label='Target distribution')
    plt.xlabel('z-score')
    plt.ylabel('Count')
    plt.xlim(-4, 4)
    plt.legend()
    plt.show()



'''
Performs time jitter test on a list of models
    models          : List of Tensorflow models
    X               : Base X array (pulses)
    Y               : Y array (delta mu)
    test            : test data
    epochs          : Number of epochs to run
    plot_jitter     : Plot the jittered pulses
    jitter          : time jitter values

    returns         : None
'''
def jitter_test(models, X, Y, test=None, epochs=100, plot_jitter=False, jitter=[0, 10, 50, 200]):

    debug_print(['running jitter test'])

    sns.set_theme(style='whitegrid')

    X_jitter = []
    X_jitter_test = []
    for j in jitter:
        x_jitter = jitter_pulses(X, j)
        if plot_jitter:
            for i in range(3):
                plt.plot(x_jitter[i, :, 0])
            plt.title("Plot of random time-translated events: ±" + str(int(j / 2) * 10) + "ns")
            plt.show()
        X_jitter.append(x_jitter)

        if test:
            x_jitter_test = jitter_pulses(test[0], j)
            X_jitter_test.append(x_jitter_test)

    losses = []
    for index, model in enumerate(models):
        for i, (x, j) in enumerate(zip(X_jitter, jitter)):
            debug_print(['model', index + 1, '/', len(models), ':', model.name, '- test', i + 1, 'of', len(jitter)])
            model = reset_weights(model)
            history = train(model, x, Y, epochs=epochs, batch_size=64, validation_split=0.2, compile=False, summary=False, callbacks=False)
            loss = history.history['val_loss']
            losses.append([j, model.name, round(loss[-1], 3)])
            if not test: linearity_plot(model, data=(x, Y))
            if test:
                x_jitter_test = X_jitter_test[i]
                linearity_plot(model, data=(x_jitter_test, test[1]), title=': ± ' + str(j // 2 * 10) + 'ns')
    df = pd.DataFrame(losses, columns=['Jitter', 'Model', 'MAE Loss: ' + str(epochs) + ' epochs'])

    '''
    if test:
        for model in models:
            for x, j in zip(X_jitter_test, jitter):
                linearity_plot(model, data=(x, Y), title=': ± ' + str(j // 2 * 10) + 'ns')
    '''

    palette = sns.color_palette("rocket_r")
    g = sns.catplot(data=df, kind='bar', x='Model', y='MAE Loss: ' + str(epochs) + ' epochs', hue='Jitter', errorbar='sd', palette=palette, alpha=0.7, height=6)
    g.figure.suptitle('Model sensitivity to temporal variation')
    g.despine(left=True)
    g.set_axis_labels('', 'Loss (MAE) after ' + '100' + ' epochs')
    g.legend.set_title('Variation (\u0394t, samples)')
    plt.show()


'''
Compare FFT model vs non-FFT model
'''
def fft_experiment(X, Y):
    X_fft = np.concatenate([np.expand_dims(fourier_decomposition(x, plot=False), axis=0) for x in tqdm(X)])
    X = np.expand_dims(X, axis=-1)
    X_fft = np.expand_dims(X_fft, axis=-1)

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

    three_layer_mlp = MLPModel(input_size=700, output_size=1, classification=False, name='3_layer_mlp')
    three_layer_mlp_2 = MLPModel(input_size=700, output_size=1, classification=False, name='3_layer_mlp')

    load_weights(three_layer_mlp)
    load_weights(three_layer_mlp_2)
    # load_weights(three_layer_mlp_1090)

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

'''
Compare models by their loss
    models          : List of Tensorflow models
    X_test          : X testing data (pulses)
    Y_test          : Y testing data (delta mu)
    X_1090_test     : X testing data, 1090 included

    returns         : None
'''
def comparison_test(models, X_test, Y_test, X_1090_test=None):
    losses = []
    for model in tqdm(models):
        x_test = X_1090_test if '1090' in model.name.split('_') else X_test
        y_test = Y_test
        load_weights(model)
        loss = tf.keras.losses.MeanAbsoluteError()(y_test, tf.transpose(model(x_test, training=False)))
        losses.append(loss)

    plt.figure(figsize=(10, 6))
    plt.bar([model.name for model in models], losses, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('Mean absolute error')
    plt.title('Model loss from reducing trainable params (on 20k samples after 200 epochs)')
    plt.show()

'''
Compare models by their training curves, metric plots, and linearity plots across a controlled number of epochs
    models          : List of Tensorflow models
    X_train         : X training data (pulses)
    Y_train         : Y training data (delta mu)
    X_test          : X testing data (pulses)
    Y_test          : Y testing data (delta mu)
'''
def model_test(models, X_train, Y_train, X_test, Y_test):
    for model in models:
        train(model, X_train, Y_train, epochs=100, batch_size=64, callbacks=False, compile=True)
        save_model_weights(model)

    for model in models:
        debug_print(['creating metric plots for', model.name])
        compare_history([model], metric='val_loss', title=model.name + ' model validation loss over training')
        linearity_plot(model, data=[np.squeeze(X_test), Y_test])

    compare_history(models, metric='val_loss', title='Model validation loss over training')

'''
Compares the history of a list of models on a particular metric
    models          : List of Tensorflow models
    metric          : Tracked model metric
'''
def compare_history(models, metric, title=None):
    for model in models:
        plt.plot(model.history.history[metric], label=model.name)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(title)
    plt.show()

'''
Evaluate the efficacy of distribution shifting
'''
def shift_distribution_test(X_train, Y_train, X_test, Y_test):
    X_train_jitter = jitter_pulses(X_train, 300)
    X_test_jitter = jitter_pulses(X_test, 300)
    three_layer_mlp = MLPModel(input_size=700, output_size=1, classification=False, name='3_layer_mlp')
    load_weights(three_layer_mlp)

    train(three_layer_mlp, X_train_jitter, Y_train, epochs=100, batch_size=64)
    linearity_plot(three_layer_mlp, (X_test_jitter, Y_test), num_samples=500, num_delta_mu=30)

    three_layer_mlp_2 = MLPModel(input_size=700, output_size=1, classification=False, name='3_layer_mlp')
    load_weights(three_layer_mlp)
    # load_weights(three_layer_mlp_2)

    X_train_shift, Y_train_shift = shift_distribution(X_train, Y_train)
    print(X_train_shift.shape)

    train(three_layer_mlp_2, X_train_shift, Y_train_shift, epochs=100)

    linearity_plot(three_layer_mlp, (X_test, Y_test))
    linearity_plot(three_layer_mlp_2, (X_test, Y_test))


'''
Compare MLP parameter sizes against time jitter performance
'''
def mlp_jitter_test(X_train, Y_train, X_test, Y_test, epochs=50):
    def num_trainable_variables(model):
        return np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])

    model_sizes = [
        [16, 1],
        [16, 16, 1],
        [16, 16, 16, 1],
        [32, 1],
        [32, 32, 1],
        [32, 32, 32, 1]
    ]

    models = [
        CustomMLPModel(input_size=700, layer_sizes=model_size) for model_size in model_sizes
    ]

    for model in models:
        model.summary()

    jitter_test(models, X_train, Y_train, (X_test, Y_test), epochs=epochs, plot_jitter=False, jitter=[0, 20, 100, 200, 300])
    

'''
Compute saliency map via integrated gradient
'''
def compute_saliency_map(model, input_sequence, baseline=None, num_steps=50, title='Saliency map of example pulse', label='MS pulse', subtitle='', save_path='figs/'):

    if baseline is None:
        baseline = np.zeros_like(input_sequence)

    scaled_inputs = [0 + (float(i) / num_steps) * (input_sequence - 0) for i in range(num_steps + 1)]
    scaled_inputs = tf.stack(tf.convert_to_tensor(scaled_inputs, dtype=tf.float32))

    with tf.GradientTape() as tape:
        tape.watch(scaled_inputs)
        predictions = model(np.expand_dims(baseline, axis=0) + scaled_inputs)
    
    grads = tape.gradient(predictions, scaled_inputs)
    integrated_gradients = (input_sequence - baseline) * grads.numpy().mean(axis=0)

    saliency_map = np.sum(integrated_gradients, axis=-1)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), gridspec_kw={'height_ratios': [1, 0.1]}, sharex='col')

    # axes[0].plot(baseline, marker='o', label='Baseline', color='blue')
    
    axes[0].plot(baseline, label='Baseline (x\')', linestyle=':')

    axes[0].plot(baseline + input_sequence, label=label, color='green')
    axes[0].set_title(title)
    axes[0].text(0.5, 0.95, subtitle, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize='smaller', color='gray')
    axes[0].legend()

    white = np.ones((10, 10))

    im = axes[1].imshow([saliency_map], cmap='viridis', aspect='auto', extent=[0, len(input_sequence), 0, 1], vmin=-5, vmax=5)
    # axes[1].set_title('Saliency Map')
    axes[1].set_yticks([])
    # cbar = plt.colorbar(white_im, ax=axes[0], orientation='vertical', fraction=0.0001, pad=0.05)
    cbar = plt.colorbar(im, ax=axes, orientation='vertical', fraction=0.008, pad=-0.02)

    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()
    plt.close()

    return saliency_map


'''
def compute_saliency_map(model, input_sequence, baseline=None, num_steps=50, title='Saliency map of example pulse', label='MS pulse', subtitle=''):
    # input_sequence = tf.convert_to_tensor(input_sequence, dtype=tf.float32)

    if baseline is None:
        baseline = tf.zeros_like(input_sequence)
    
    scaled_inputs = [baseline + (float(i) / num_steps) * (input_sequence - baseline) for i in range(num_steps + 1)]
    scaled_inputs = tf.stack(tf.convert_to_tensor(scaled_inputs))

    for i in scaled_inputs.numpy():
        plt.plot(i[:, 0])
    plt.show()

    with tf.GradientTape() as tape:
        tape.watch(scaled_inputs)

        predictions = model(scaled_inputs)
    gradients = tape.gradient(predictions, scaled_inputs)
    print(gradients)
    saliency_map = tf.reduce_mean(gradients, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), gridspec_kw={'height_ratios': [1, 0.1]}, sharex='col')

    # axes[0].plot(baseline, marker='o', label='Baseline', color='blue')
    
    axes[0].plot(baseline, label='Baseline (x\')', linestyle=':')

    axes[0].plot(input_sequence, label=label, color='green')
    axes[0].set_title(title)
    axes[0].text(0.5, 0.95, subtitle, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize='smaller', color='gray')
    axes[0].legend()

    white = np.ones((10, 10))
    white_im = axes[1].imshow(white, cmap='gray', aspect='auto', extent=[0, 1, 0, 1], vmax=1)

    im = axes[1].imshow([saliency_map], cmap='viridis', aspect='auto', extent=[0, len(input_sequence), 0, 1], vmin=-100, vmax=100)
    # axes[1].set_title('Saliency Map')
    axes[1].set_yticks([])
    # cbar = plt.colorbar(white_im, ax=axes[0], orientation='vertical', fraction=0.0001, pad=0.05)
    cbar = plt.colorbar(im, ax=axes, orientation='vertical', fraction=0.008, pad=-0.02)

    plt.tight_layout()
    plt.savefig('figs/' + title + ' ' + subtitle)
    # plt.show()
    plt.clf()

    return saliency_map
'''


'''
Plot parameter/peformance tradeoff from keras tuner
'''
def plot_parameter_performance(paths, lim=200, title='Number of Parameters vs. Training Performance'):
    parameters = []
    losses = []
    colors = []
    layer_sizes = []
    for path in paths:
        for dir in os.listdir(path):
            num_parameters = 0
            num_previous = 700
            if os.path.isfile(path + dir): continue
            with open(path + dir + '/trial.json') as f:
                trial = json.load(f)
                loss = trial['score']
                try:
                    if loss > lim: continue
                    layers = trial['hyperparameters']['values'].values()
                    layer_sizes.append(list(layers))
                    for value in layers:
                        num_parameters += num_previous * value + value
                        num_previous = value
                    num_parameters += num_previous + 1
                    parameters.append(num_parameters)
                    losses.append(loss)
                    colors.append(sum([1 if layer > 1 else 0 for layer in layers]))
                except: continue
    fig, ax = plt.subplots()
    sc = plt.scatter(parameters, losses, c=colors, lw=2)# , label=str(i + 1) + ' layers')

    annot = ax.annotate("", xy=(0,0), xytext=(10,10),textcoords='offset points',
                        bbox=dict(boxstyle='round', fc='gray'),
                        arrowprops=dict(arrowstyle='-'))
    annot.set_visible(False)

    def update_annotation(ind):
        pos = sc.get_offsets()[ind['ind'][0]]
        annot.xy = pos
        text = ''
        for i in ind['ind']:
            text += 'Layers: ' + '->'.join(map(str, layer_sizes[i])) + '\n'
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.6)
        

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annotation(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    
    plt.xscale('log')
    plt.title(title)
    plt.legend()
    plt.show()

    return fig


'''
Determine output as a function of jitter
'''
def jitter_function_plot(model, X_train, Y_train, X_test, Y_test):
    X_train_jitter = jitter_pulses(X_train, t=400)
    X_test_jitter = jitter_pulses(X_test, t=400)
    # train(three_layer_mlp, X_train_jitter, Y_train, epochs=25, batch_size=32)

    # save_model_weights(model)
    # load_weights(model)
    train(model, X_train_jitter, Y_train, epochs=100, batch_size=64)

    jitters = np.zeros(300 // 5)
    delta_mu_vs_jitter = np.zeros((200, 300 // 5))
    for i in tqdm(range(200)):
        for j, t in enumerate(range(-150, 150, 5)):
            jitters[j] = t
            x = slide_pulse(X_test[i], -t)
            delta_mu = Y_test[i]
            pred_delta_mu = model(np.expand_dims(x, axis=0))
            delta_mu_vs_jitter[i][j] = int(pred_delta_mu) - delta_mu
    
    mean_delta_mu_vs_jitter = np.mean(delta_mu_vs_jitter, axis=0)
    err = np.std(delta_mu_vs_jitter, axis=0)
    plt.fill_between(jitters, mean_delta_mu_vs_jitter-err, mean_delta_mu_vs_jitter+err, color=(0.1, 0.2, 0.5, 0.3))
    plt.plot(jitters, mean_delta_mu_vs_jitter)
        
    plt.title('Jitter effect on predicted delta mu deviation (16/1) [3/6/24]')
    plt.ylabel('Deviation from true delta mu')
    plt.xlabel('Jitter amount')
    plt.show()


'''
Plot saliency map as a function of jitter
'''
def saliency_map_jitter(model, X_train, Y_train, X_test, Y_test, X_dist):
    X_train_jitter = jitter_pulses(X_train, t=400)
    train(model, X_train_jitter, Y_train, epochs=100, batch_size=64)
    for t in tqdm(range(-150, 150, 10)):
        x = slide_pulse(X_test[0], -t)
        delta_mu = Y_test[0]
        pred_delta_mu = model(np.expand_dims(x, axis=0))
        compute_saliency_map(
            model, 
            x, 
            baseline=slide_pulse(X_dist, -t),
            title=f'[3-1-23] Saliency map of example pulse: Pred delta mu = {int(pred_delta_mu[0][0])}ns',
            label=f'MS pulse (delta mu = {int(delta_mu)}ns)',
            subtitle=model.name,
            save_path='gif/' + str(t + 150) + '.png'
        )

    convert_files_to_gif('gif/', 'saliency_map.gif')

'''
Evaluate performance of model w.r.t. amount of training data
'''
def performance_vs_data(model, data_sizes, X_train, Y_train, X_test, Y_test):
    all_losses = []
    final_losses = []
    for data_size in data_sizes:
        sz = int(data_size * 0.8)
        history = train(model, X_train[:sz], Y_train[:sz], epochs=200, batch_size=256)
        losses = history.history['val_loss']
        final_losses.append(losses[-1])
        all_losses.append(losses)
        linearity_plot(model, (X_test, Y_test), title=' ' + str(data_size) + ' ')
        reset_weights(model)

    for data_size, losses in zip(data_sizes, all_losses):
        plt.plot(losses, label='data size ' + str(data_size))
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.ylim(100, 150)
    plt.legend()
    plt.title(model.name + ' Training Convergence vs. Data Size')
    plt.show()
    
    plt.plot(final_losses)
    plt.xlabel('Data Size')
    plt.ylabel('Loss (MAE)')
    plt.title(model.name + ' Training Loss vs. Data Size')
    plt.show()


'''
Compare accuracy/data representations for different autoencoder sizes
'''
def compare_latent_dim_compression(latent_sizes, X_train, Y_train, X_test, Y_test):
    for latent_size in latent_sizes:
        autoencoder = Autoencoder(input_size=700, encoder_layer_sizes=[512, 128, latent_size], decoder_layer_sizes=[128, 700])

        train(autoencoder, X_train, X_train, epochs=200, batch_size=256)

        for x in X_test[:2]:
            plt.figure(figsize=(6, 5))
            plt.plot(x, label='Original Pulse')
            plt.plot(autoencoder(np.array([x]))[0], label='Reconstructed Pulse')
            # plt.title('Pulse reconstruction: ' + autoencoder.name)
            plt.xlabel('Sample [10 ns]')
            plt.ylabel('Signal amplitude [phd/10 ns]')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('autoencoder_reconstruction_high_res.png', dpi=300)
            plt.show()

        encoded_X_train = autoencoder.encode(X_train)
        encoded_X_test = autoencoder.encode(X_test)

        debug_print(['Size before compression:', sys.getsizeof(X_train), 'bytes\n',
                    'Size after compression:', sys.getsizeof(X_train) // 700 * latent_size, f'bytes, ({700 // latent_size}x reduction)'])
        regression_model_small = CustomMLPModel(input_size=latent_size, layer_sizes=[256, 128, 32, 1])

        train(regression_model_small, encoded_X_train, Y_train, epochs=100, batch_size=256)

        linearity_plot(regression_model_small, (encoded_X_test, Y_test))

'''
Compare accuracy/data representations for different autoencoder sizes with arrival times
'''
def compare_latent_dim_compression_at(latent_sizes, X_train, Y_train, AT_train, X_test, Y_test, AT_test):
    for latent_size in latent_sizes:
        at_weight = 1e-7
        autoencoder = MultiHeaddedAutoencoder(
            input_size=700, 
            encoder_layer_sizes=[512, 256, latent_size], 
            decoders=[[256, 512, 700], [128, 128, 148]],
            loss_weights=[1 - at_weight, at_weight])

        train(autoencoder, X_train, [X_train, AT_train], epochs=50, batch_size=256)

        for x, dmu, at in zip(X_test[:2], Y_test[:2], AT_test[:2]):
            x_hat, at_hat = autoencoder(np.array([x]))
            plt.plot(x, label='Original Pulse')
            plt.plot(x_hat[0], label='Reconstructed Pulse')
            plt.title('Pulse reconstruction: ' + autoencoder.name)
            plt.legend()
            plt.show()

            plt.title(f'True vs predicted electron arrival times for Δμ={dmu} pulse')
            plt.plot(at, marker='+', label='True electron arrival times')
            plt.plot(at_hat[0], marker='o', label='Predicted electron arrival times')
            plt.ylabel('Arrival time (samples, 10ns)')
            plt.xlabel('Electron number')
            plt.legend()
            plt.show()

        continue


        encoded_X_train = autoencoder.encode(X_train)
        encoded_X_test = autoencoder.encode(X_test)

        debug_print(['Size before compression:', sys.getsizeof(X_train), 'bytes\n',
                    'Size after compression:', sys.getsizeof(X_train) // 700 * latent_size, f'bytes, ({700 // latent_size}x reduction)'])
        regression_model_small = CustomMLPModel(input_size=latent_size, layer_sizes=[256, 128, 32, 1])

        train(regression_model_small, encoded_X_train, Y_train, epochs=100, batch_size=256)

        linearity_plot(regression_model_small, (encoded_X_test, Y_test))

'''
Compare relative deviation or other data-specific methods by compression performance
'''
def compare_data_compression(autoencoder, latent_size, X_train, Y_train):
    base_autoencoder = copy.deepcopy(autoencoder)
    test_autoencoder = copy.deepcopy(autoencoder)

    X_dev, X_params, epsilon = get_relative_deviation(X_train)

    train(test_autoencoder, X_dev, X_dev, epochs=100, batch_size=128)
    train(base_autoencoder, X_train, X_train, epochs=100, batch_size=128)
    
    dev_encoded_X_train = test_autoencoder.encode(X_train)
    encoded_X_train = base_autoencoder.encode(X_train)

    base_model = CustomMLPModel(input_size=latent_size, layer_sizes=[256, 128, 32, 1])
    test_model = CustomMLPModel(input_size=latent_size, layer_sizes=[256, 128, 32, 1])

    train(base_model, dev_encoded_X_train, Y_train, epochs=100, batch_size=256)

    train(test_model, encoded_X_train, Y_train, epochs=100, batch_size=256)

    # plot
    for dev, params, x in zip(X_dev[:2], X_params[:2], X_train[:2]):
        dev = test_autoencoder(np.array([dev]))[0]
        dev = np.reshape(dev, (-1, 1))
        a = np.linspace(0, dev.shape[0], dev.shape[0])
        fit = np.expand_dims(gaussian(a, *params), axis=-1) + np.ones_like(dev) * epsilon
        x_r = dev * fit
        plt.title('Gaussian deviation autoencoder')
        plt.plot(x, label='Original Pulse')
        plt.plot(x_r, label='Reconstructed Pulse')
        plt.xlabel('Samples')
        plt.show()

    
    for x in X_train[:2]:
        x_r = base_autoencoder(np.array([x]))[0]
        plt.title('Vanilla autoencoder')
        plt.plot(x, label='Original Pulse')
        plt.plot(x_r, label='Reconstructed Pulse')
        plt.xlabel('Samples')
        plt.show()

    linearity_plot(base_model, (dev_encoded_X_train, Y_train))

    linearity_plot(test_model, (encoded_X_train, Y_train))


'''
Area fraction vs. performance
'''
def area_fraction_test(model, X, Y, areafrac):
    num_areafracs = 10

    areafrac_indices = np.zeros(num_areafracs, dtype=np.int32)
    areafrac_data = np.empty((num_areafracs, int(X.shape[0] / (num_areafracs * 1.5))), dtype=np.int32)
    print(areafrac_data.shape, X.shape, areafrac.shape)
    for i, frac in enumerate(areafrac):
        af = 0.5 - np.abs(0.5 - frac) - 1e-4
        index = int(np.floor(af * 2 * num_areafracs))
        if areafrac_indices[index] >= areafrac_data.shape[1]: continue
        areafrac_data[index][areafrac_indices[index]] = i
        areafrac_indices[index] += 1

    model = CustomMLPModel(input_size=700, layer_sizes=[256, 128, 32, 1])

    areafrac_models = [CustomMLPModel(input_size=700, layer_sizes=[256, 128, 32, 1]) for _ in range(num_areafracs)]

    for areafrac_model, indices in zip(areafrac_models, areafrac_data):
        X_areafrac = X[indices]
        Y_areafrac = Y[indices]
        train(areafrac_model, X_areafrac, Y_areafrac, epochs=200, batch_size=128)

    train(model, X, Y, epochs=50, batch_size=128)

    for areafrac_model, indices in zip(areafrac_models, areafrac_data):
        X_areafrac = X[indices][:100]
        Y_areafrac = Y[indices][:100]
        a = areafrac[indices][:100]
        af = 0.5 - np.abs(0.5 - a)
        Y_pred_areafrac = areafrac_model(X_areafrac)
        error = np.abs(Y_areafrac - Y_pred_areafrac[:, 0])
        plt.scatter(af, error, s=5)
        plt.plot(np.unique(af), np.poly1d(np.polyfit(af, error, 1))(np.unique(af)))
    
    X_test = X[:2000]
    Y_test = Y[:2000]
    areafrac_test = areafrac[:2000]
    af_test = 0.5 - np.abs(0.5 - areafrac_test)
    Y_pred = model(X_test)[:, 0]
    error = np.abs(Y_test - Y_pred)
    plt.plot(np.unique(af_test), np.poly1d(np.polyfit(af_test, error, 1))(np.unique(af_test)), label='All areafrac model')

    plt.legend()
    plt.title('Separately trained areafrac model performance')
    plt.ylabel('MAE')
    plt.xlabel('Area fraction')
    plt.show()


def electron_arrival_time_test(X, AT):
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


def electron_counts_test(electron_counts=(1, 2, 4, 8, 16, 32, 64, 128, 256)):
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


def autoencoder_test(X, Y, AT):
    at_weight = 1e-7
    mhae = MultiHeaddedAutoencoder(
        input_size=700, 
        encoder_layer_sizes=[512, 256, 16], 
        decoders=[[256, 512, 700], [128, 128, 148]],
        loss_weights=[1 - at_weight, at_weight])
    
    # train(mhae, X, [X, AT], epochs=20, batch_size=128, summary=True)

    # compare_latent_dim_compression_at([1, 2, 4, 8, 16, 32, 64, 128, 256], X, Y, AT, X[:100], Y[:100], AT[:100])

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
        
        at_hist, _ = np.histogram(at, bins=np.arange(0, 700, 1))
        at_hat_hist, _ = np.histogram(at_hat, bins=np.arange(0, 700, 1))
        plot_at_hists(at_hist, label='True arrival times')
        plot_at_hists(at_hat_hist, label='Predicted arrival times')
        plt.title('True vs predicted histogram of electron arrival times')
        plt.xlabel('Arrival time (samples, 10ns)')
        plt.ylabel('Number of electrons')
        plt.legend()
        plt.show()


def arrival_time_test(X, Y, AT):
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


def arrival_time_model_test(X, XC, Y, AT):
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

        at_hat_hist, _ = np.histogram(at_hat, bins=np.arange(0, 700, 1))
        at_hist, _ = np.histogram(at, bins=np.arange(0, 700, 1))
        plot_at_hists(at_hat_hist, label='Predicted arrival times')
        plot_at_hists(at_hist, label='True arrival times')
        plt.title('True vs predicted histogram of electron arrival times')
        plt.xlabel('Arrival time (samples, 10ns)')
        plt.ylabel('Number of electrons')
        plt.legend()
        plt.show()

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


def categorical_mse(y, y_hat, categories=(0, 1, 2, 3, 4)):
    mse_values = []

    for cat in categories:
        indices = (y == cat)
        mse = np.mean((y_hat[indices] - y[indices]) ** 2)
        mse_values.append(mse)

    return mse_values

def plot_mse_histogram(y, y_hat, categories=(0, 1, 2, 3, 4), title='Baseline MSE for each photon count [4-25-24]', save_path=''):
    mse_values = categorical_mse(y, y_hat, categories)
    bins = np.array(categories + (5,))
    
    plt.stairs(mse_values, bins - 0.5)
    plt.xticks(np.arange(len(categories)))
    plt.xlabel('Photon count')
    plt.ylabel('MSE')
    plt.ylim((0, 16))
    plt.title(title)
    if save_path == '':
        plt.show()
    else:
        plt.savefig(save_path)
        plt.clf()


def photon_count_check(PXC_FLAT):
    counts, bins = np.histogram(PXC_FLAT)
    counts, bins = counts[:5], bins[:6]
    plt.stairs(counts / len(PXC_FLAT), bins - 0.5)
    for i in range(len(counts)):
        plt.text(bins[i], counts[i] / len(PXC_FLAT) + 0.01, str(round(counts[i] / len(PXC_FLAT), 4)), ha='center')
    
    plt.xlabel('Photon count')
    plt.ylabel('Proportion of dataset')
    plt.title('Proportion of each photon count in simulated SS channel pulse')
    plt.show()


def delta_test(num_samples, XC, PXC, XC_FLAT, PXC_FLAT, window_size, deltas=(1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2), num_tests=100):
    sample_indices = np.array([i for i in range(num_samples)])
    np.random.shuffle(sample_indices)
    X_chunks, Y_chunks = get_windowed_data(XC_FLAT[:num_samples + window_size], PXC_FLAT[:num_samples + window_size], window_size)
    
    for delta in deltas:
        debug_print(['Delta:', delta])

        photon_model = BaselinePhotonModel(input_size=window_size, layer_sizes=[32, 32, 1], max=4, delta=delta)
        train(photon_model, X_chunks[sample_indices], Y_chunks[sample_indices], epochs=50, batch_size=512)
        y_hat = photon_model(X_chunks[:100000])
        y = Y_chunks[:100000]
        plot_mse_histogram(y, y_hat, title=f'window={window_size} model MSE for each photon count [5-03-24]', save_path=f'{window_size}_gram_pe_smse')

        num_tests = 100
        diffs = np.empty((num_tests, 700 - window_size))
        for i in range(num_tests):
            x, y = np.transpose(np.reshape(XC[i], (16*16, 700))), np.transpose(np.reshape(PXC[i], (16*16, 700)))
            y_ = np.empty_like(y[window_size//2:-window_size//2], dtype=np.float32)
            for c in range(16*16):
                x_chunks, y_chunks = get_windowed_data(x[:, c], y[:, c], window_size)
                y_[:, c] = photon_model(x_chunks)[:, 0]
            y = y[window_size//2:-window_size//2]
            diffs[i] = np.sum(y - np.round(y_, decimals=0), axis=1)
            
            if i < 5:
                plt.figure(figsize=(8, 6))
                plt.plot(np.cumsum(np.sum(y, axis=(1))), label='True summed pulse CDF')
                # plt.plot(np.cumsum(np.sum(y_, axis=(1))), label='Predicted summed pulse CDF', color='orange')
                plt.plot(np.cumsum(np.sum(np.round(y_, decimals=0), axis=1)), label='Predicted summed pulse (rounded) CDF', color='red', alpha=0.5) 

                plt.xlabel('Sample')
                plt.ylabel('Photon count')
                plt.title(f'delta={delta} SS pulse summed photon count prediction')
                plt.legend()
                # plt.show()
                plt.savefig(f'window_{window_size}_delta_{int(delta * 10)}_{i}_pulse_summed_smse_pe_cdf')
                plt.clf()
                plt.close()


                channel = 36
                plt.figure(figsize=(8, 6))
                plt.plot(np.cumsum(y[:, channel]), label=f'Channel {channel} pulse CDF')
                # plt.plot(np.cumsum(y_[:, channel]), label=f'Predicted channel pulse CDF')
                plt.plot(np.cumsum(np.round(y_[:, channel], decimals=0)), label=f'Predicted channel pulse (rounded) CDF')
                plt.xlabel('Sample')
                plt.ylabel('Photon count')
                plt.title(f'delta={delta} single channel photon count prediction')
                plt.legend()
                plt.savefig(f'window_{window_size}_delta_{int(delta * 10)}_{i}_pulse_smse_pe_cdf')
                plt.clf()
                plt.close()
                # plt.show()
        
        stds = np.std(diffs, axis=0)
        means = np.mean(diffs, axis=0)

        plt.figure(figsize=(8, 6))
        plt.errorbar(np.arange(stds.shape[0]), means, yerr=stds, capsize=0, label='Error', color='#0e68cf', alpha=0.3)
        plt.plot(means, label='Mean error', color='#0e68cf')
        plt.xlabel('Sample')
        plt.ylabel('Photon count')
        plt.title(f'delta={delta} model error for each photon count [5-3-24]')
        plt.legend()
        plt.savefig(f'window_{window_size}_delta_{int(delta * 10)}_mse_pe')
        plt.clf()
        plt.close()


def channel_photon_size_test(num_samples, XC, PXC, XC_FLAT, PXC_FLAT, window_size, delta=1.35):

    sample_indices = np.array([i for i in range(num_samples)])
    np.random.shuffle(sample_indices)
    X_chunks, Y_chunks = get_windowed_data(XC_FLAT[:num_samples + window_size], PXC_FLAT[:num_samples + window_size], window_size)

    photon_model = BaselinePhotonModel(input_size=window_size, layer_sizes=[32, 32, 1], max=4, delta=delta)
    train(photon_model, X_chunks[sample_indices], Y_chunks[sample_indices], epochs=100, batch_size=512)
    y_hat = photon_model(X_chunks[:100000])
    y = Y_chunks[:100000]
    plot_mse_histogram(y, y_hat, title=f'window={window_size} model MSE for each photon count [5-03-24]', save_path=f'{window_size}_gram_pe_smse')

    num_tests = 1000
    diffs = np.empty((num_tests, 700 - window_size))
    uncompressed_bits, compressed_bits = 0, 0
    for i in tqdm(range(num_tests)):
        x, y = np.transpose(np.reshape(XC[i], (16*16, 700))), np.transpose(np.reshape(PXC[i], (16*16, 700)))
        y_ = np.empty_like(y[window_size//2:-window_size//2], dtype=np.float32)
        for c in range(16*16):
            x_chunks, y_chunks = get_windowed_data(x[:, c], y[:, c], window_size)
            y_[:, c] = photon_model(x_chunks)[:, 0]
        y = y[window_size//2:-window_size//2]
        diffs[i] = np.sum(y - np.round(y_, decimals=0), axis=1)

        uncompressed_bits += np.sum(x > 1e-6) * 14
        compressed_bits += np.sum(y_ > 1e-6) * 2

    
    error = np.mean(np.abs(diffs))
    debug_print_(f'Mean error: {error : .4f} photons per pulse')
    debug_print_(f'Compression ratio: {uncompressed_bits / compressed_bits : .4f}')
    


def channel_photon_test(num_samples, XC, PXC, XC_FLAT, PXC_FLAT, window_size):
    X_chunks, Y_chunks = get_windowed_data(XC_FLAT[:num_samples + window_size], PXC_FLAT[:num_samples + window_size], window_size)

    sample_indices = np.array([i for i in range(num_samples)])
    np.random.shuffle(sample_indices)
    photon_model = BaselinePhotonModel(input_size=window_size, layer_sizes=[32, 32, 1], max=4, delta=1.3)
    train(photon_model, X_chunks[sample_indices], Y_chunks[sample_indices], epochs=0, batch_size=512)
    y_hat = photon_model(X_chunks[:100000])
    y = Y_chunks[:100000]

    num_tests = 10000
    errors = np.empty((num_tests, 16, 16, 700 - window_size))
    photon_counts = np.zeros((num_tests, 16, 16))
    for test in tqdm(range(num_tests)):
        for i in range(16):
            for j in range(16):
                x_event, y_event = np.reshape(XC[test][i][j], (-1, 700)), np.reshape(PXC[test][i][j], (700,))
                photon_counts[test, i, j] = np.sum(y_event)
                print(photon_counts[test])
                x_event_chunks, y_event_chunks = get_windowed_data(x_event, y_event, window_size)
                print(x_event_chunks.shape, x_event_chunks)
                y_ = photon_model(x_event_chunks)[:, 0]
                y = y[window_size//2:-window_size//2]
                errors[test, i, j] = y - np.round(y_, decimals=0)
    
    max_count = int(max(photon_counts))
    photon_count_mean_errors = np.zeros(max_count)
    for test in range(num_tests):
        photon_count = photon_counts[test]
        photon_count_mean_errors[photon_count] += np.mean(errors[test])

    photon_count_mean_errors /= np.bincount(photon_counts)
    plt.plot(photon_count_mean_errors)
    plt.xlabel('Photon count')
    plt.ylabel('Mean error')
    plt.title('Mean error of channel photon model in 100k SS pulses')
    plt.show()
    

def plot_hit_pattern(hit, filename='hit_pattern'):
    dprint('plotting hit pattern')
    image_frames = []
    imgs = np.transpose(hit, axes=[2, 0, 1])
    for t in tqdm(imgs):
        plt.imshow(t, vmin=0, vmax=1)
        plt.title('Hit pattern')
        
        plt.gcf().canvas.draw()
        width, height = plt.gcf().get_size_inches() * plt.gcf().dpi
        data = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
        image_array = data.reshape(int(height), int(width), 3)

        image_frame = Image.fromarray(image_array)
        image_frames.append(image_frame)

    datetime_tag = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    image_frames[0].save(filename + f'_{datetime_tag}.gif', 
                        save_all = True, 
                        duration = 20,
                        loop = 0,
                        append_images = image_frames[1:])
    

def plot_hit_comparison(hit1, hit2, filename='hit_comparison'):
    dprint('plotting hit comparison')
    image_frames = []
    hit1, hit2 = np.transpose(hit1, axes=[2, 0, 1]), np.transpose(hit2, axes=[2, 0, 1])
    for t1, t2 in tqdm(zip(hit1, hit2)):
        t = np.concatenate((t1, t2), axis=1)
        plt.imshow(t, vmin=0, vmax=1)
        plt.title('Hit comparison')
        
        plt.gcf().canvas.draw()
        width, height = plt.gcf().get_size_inches() * plt.gcf().dpi
        data = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
        image_array = data.reshape(int(height), int(width), 3)

        image_frame = Image.fromarray(image_array)
        image_frames.append(image_frame)
    
    datetime_tag = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    image_frames[0].save(filename + f'_{datetime_tag}.gif',
                         save_all = True,
                         duration = 20,
                         loop = 0,
                         append_images = image_frames[1:])


def graph_electron_arrival_prediction(XC, AT_hist, epochs=100):
    graph_adjacency_matrix = create_grid_adjacency(XC.shape[1])
    XC_in = XC.reshape(XC.shape[0], -1, 700)
    
    graph_model = GraphElectronModel(
        adjacency_matrix    = graph_adjacency_matrix,
        graph_layer_sizes   = [512, 256, 128, 64, 32, 16, 8, 4], 
        layer_sizes         = [512, 700],
        loss                = ScaledMeanSquaredError(delta=1.3)
    )

    if epochs > 0:
        train(graph_model, XC_in, AT_hist, epochs=epochs, batch_size=128, plot_history=True)

    for i in range(30):
        plot_at_hist(AT_hist[i], label='True electron arrivals')
        at_hist_hat = graph_model(np.expand_dims(XC_in[i], axis=0))[0]
        plot_at_hist(at_hist_hat, label='Predicted electron arrivals')
        plt.legend()
        plt.show()


def graph_channel_electron_arrival_prediction(XC, at_channel_hist, epochs=100):
    graph_adjacency_matrix = create_grid_adjacency(XC.shape[1])
    XC_in = XC.reshape(XC.shape[0], -1, 700)

    graph_model = GraphElectronModel(
        adjacency_matrix    = graph_adjacency_matrix,
        graph_layer_sizes   = [700, 700, 700], 
        layer_sizes         = None
    )
    # graph_model = MLPElectronModel()
    
    graph_model = load('graph_channel_model_2')
    graph_model.build((None, 256, 700))
    graph_model.compile(loss=MeanSquaredEMDLoss3D(), optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4))
    graph_model.summary()

    # plot_hit_comparison(at_channel_hist[0], XC[0])
    at_channel_hist = at_channel_hist.reshape(at_channel_hist.shape[0], -1, 700)

    if epochs > 0:
        train(graph_model, XC_in, at_channel_hist, epochs=epochs, batch_size=128, plot_history=True)

    # save_model(graph_model, 'graph_channel_model_2')

    x_test = XC_in[:5000]
    y_test = at_channel_hist[:5000]
    y_hat_test = graph_model(x_test)
    y_hat_test = np.array(y_hat_test, dtype=np.float32)
    y_test_cumsum = np.cumsum(y_test, axis=2)
    y_hat_test_cumsum = np.cumsum(y_hat_test, axis=2)
    test_cumsum_error = np.reshape(np.abs(y_test_cumsum - y_hat_test_cumsum), (5000 * 256, 700))
    cumsum_error_16_84 = np.percentile(test_cumsum_error, [16, 84], axis=0)
    mean_total_electrons = np.mean(y_test_cumsum, axis=(0, 1))

    plt.fill_between(np.arange(700), cumsum_error_16_84[0], cumsum_error_16_84[1], alpha=0.5, label='16-84% error')
    plt.plot(np.mean(test_cumsum_error, axis=0), label='Mean error')
    plt.plot(mean_total_electrons, label='Mean electron count')
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('Number of electrons')
    plt.title('Mean channel-level cumulative sum error for each sample')
    plt.show()
    

    for i in range(5):
        y = np.array(at_channel_hist[i], dtype=np.float32)
        y = np.reshape(y, (1, 16, 16, 700))
        y_cumsum = np.cumsum(y, axis=3)
        y_cumsum_x = tf.cumsum(y_cumsum, axis=1)
        y_cumsum_xy = tf.cumsum(y_cumsum_x, axis=2)[0]

        y_hat = np.array(graph_model(np.expand_dims(XC_in[i], axis=0)), dtype=np.float32)
        y_hat = np.reshape(y_hat, (1, 16, 16, 700))
        y_hat_cumsum = np.cumsum(y_hat, axis=3)
        y_hat_cumsum_x = tf.cumsum(y_hat_cumsum, axis=1)
        y_hat_cumsum_xy = tf.cumsum(y_hat_cumsum_x, axis=2)[0]

        loss = np.mean(np.abs(y_cumsum_xy - y_hat_cumsum_xy))

        concat_y_yhat = np.concatenate((y_cumsum_xy, y_hat_cumsum_xy), axis=0)

        plt.title(f'Loss: {loss : .4f}')
        plt.imshow(concat_y_yhat[:, :, -1])
        plt.colorbar()
        plt.show()

    for i in range(5):
        at_channel_hist_i = np.array(at_channel_hist[i], dtype=np.float32)
        at_channel_hist_hat_i = np.array(graph_model(np.expand_dims(XC_in[i], axis=0))[0], dtype=np.float32)

        y = tf.reshape(at_channel_hist_i, (16, 16, 700))
        y_hat = tf.reshape(at_channel_hist_hat_i, (16, 16, 700))

        y_cumsum = tf.cumsum(y, axis=2)
        y_cumsum_x = tf.cumsum(y_cumsum, axis=0)
        y_cumsum_xy = tf.cumsum(y_cumsum_x, axis=1)

        y_hat_cumsum = tf.cumsum(y_hat, axis=2)
        y_hat_cumsum_x = tf.cumsum(y_hat_cumsum, axis=0)
        y_hat_cumsum_xy = tf.cumsum(y_hat_cumsum_x, axis=1)

        y_cumsum_sum = tf.reduce_sum(y_cumsum, axis=[0, 1])
        y_hat_cumsum_sum = tf.reduce_sum(y_hat_cumsum, axis=[0, 1])

        channel_mse = tf.reduce_mean(tf.square(y_cumsum_xy - y_hat_cumsum_xy))
        sum_mse = tf.reduce_mean(tf.square(y_cumsum_sum - y_hat_cumsum_sum))

        dprint(f'Channel MSE: {channel_mse : .4f}, Sum MSE: {sum_mse : .4f}')

        # if i == 0:
        #     plot_hit_comparison(y, y_hat)

        for j in np.random.permutation(256)[:20]:
            at_channel_j_hist_i = np.cumsum(at_channel_hist_i[j])
            at_channel_j_hist_hat_i = np.cumsum(at_channel_hist_hat_i[j])

            plt.title(f'Channel {j} liquid electron arrival time histogram')
            plot_at_hist(at_channel_j_hist_i, label=f'True electron arrivals')
            plot_at_hist(at_channel_j_hist_hat_i, label=f'Predicted electron arrivals')
            plt.ylim(0, 4)
            plt.legend()
            plt.show()

    for i in range(10):
        at_hist = np.sum(at_channel_hist[i], axis=0)
        plot_at_hist(at_hist, label='True electron arrivals')
        at_channel_hist_hat = graph_model(np.expand_dims(XC_in[i], axis=0))[0]
        at_hist_hat = np.sum(at_channel_hist_hat, axis=0)
        plot_at_hist(at_hist_hat, label='Predicted electron arrivals')
        plt.legend()
        plt.show()

        plot_at_hist(np.cumsum(at_hist), label='True electron arrivals (CDF)')
        plot_at_hist(np.cumsum(at_hist_hat), label='Predicted electron arrivals (CDF)')
        plt.legend()
        plt.show()


def graph_electron_arrival_prediction(XC, at_channel_hist, epochs=100, savefigs=True):
    # plot_hit_comparison(at_channel_hist[0], XC[0])
    AT = np.sum(at_channel_hist, axis=(1, 2))

    graph_adjacency_matrix = create_grid_adjacency(XC.shape[1])
    A_sparse = tf.sparse.from_dense(graph_adjacency_matrix)
    XC_in = XC.reshape(XC.shape[0], -1, 700)

    graph_model = GraphElectronModel(
        adjacency_matrix    = graph_adjacency_matrix,
        graph_layer_sizes   = [700, 256, 32], 
        layer_sizes         = [512, 700],
    )

    # graph_model = load('graph_model')
    graph_model.build((None, 256, 700))
    graph_model.compile(loss=MeanAbsoluteEMDLoss(), optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    graph_model.summary()

    if epochs > 0:
        train(graph_model, XC_in, AT, epochs=epochs, batch_size=128, plot_history=False)

    save_model(graph_model, 'graph_model')

    x_test, y_test = XC_in[:5000], AT[:5000]
    y_hat_test = graph_model(x_test)
    y_hat_test = np.array(y_hat_test, dtype=np.float32)
    y_test_cumsum = np.cumsum(y_test, axis=1)
    y_hat_test_cumsum = np.cumsum(y_hat_test, axis=1)
    test_cumsum_error = np.abs(y_test_cumsum - y_hat_test_cumsum)
    cumsum_error_16_84 = np.percentile(test_cumsum_error, [16, 84], axis=0)
    mean_total_electrons = np.mean(y_test_cumsum, axis=0)

    plt.fill_between(np.arange(700), cumsum_error_16_84[0], cumsum_error_16_84[1], alpha=0.5, label='16-84% error')
    plt.plot(np.mean(test_cumsum_error, axis=0), label='Mean error')
    # plt.plot(mean_total_electrons, label='Mean electron count')
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('Number of electrons')
    plt.title('Mean cumulative sum error for each sample')
    if savefigs:
        plt.savefig(datetime_tag + '_graph_model_cumsum_error')
    else:
        plt.show()

    counts = np.sum(at_channel_hist, axis=(1, 2, 3))
    # Plot error by number of electrons in pulse
    errors_by_counts = np.zeros((counts.shape[0], np.max(counts) + 1))
    count_by_counts = np.zeros((counts.shape[0], np.max(counts) + 1))
    for i in tqdm(range(0, XC.shape[0], 5000)):
        x_test, y_test = XC_in[i:i+5000], AT[i:i+5000]
        y_hat_test = graph_model(x_test)
        y_hat_test = np.array(y_hat_test, dtype=np.float32)
        y_test_cumsum = np.cumsum(y_test, axis=1)
        y_hat_test_cumsum = np.cumsum(y_hat_test, axis=1)
        test_cumsum_error = np.abs(y_test_cumsum - y_hat_test_cumsum)
        for count, error in zip(counts, test_cumsum_error):
            errors_by_counts[count] += np.sum(error)
            count_by_counts[count] += 1
    errors_by_counts /= count_by_counts
    plt.plot(errors_by_counts)
    plt.xlabel('Number of electrons in pulse')
    plt.ylabel('Mean absolute EMD error')
    plt.title('Mean error by number of electrons in pulse')
    if savefigs:
        plt.savefig(datetime_tag + '_graph_model_error_by_count')
    else:
        plt.show()

    
    for i in range(10):
        at_i = np.array(AT[i], dtype=np.float32)
        at_hat_i = np.array(graph_model(np.expand_dims(XC_in[i], axis=0))[0], dtype=np.float32)

        y_cumsum = tf.cumsum(at_i, axis=0)
        y_hat_cumsum = tf.cumsum(at_hat_i, axis=0)

        sum_mse = tf.reduce_mean(tf.square(y_cumsum - y_hat_cumsum))

        dprint(f'EMD MSE: {sum_mse : .4f}')

        plt.title(f'Liquid electron arrival time histogram')
        plot_at_hist(at_i, label=f'True electron arrivals')
        plot_at_hist(at_hat_i, label=f'Predicted electron arrivals')
        plt.legend()
        if savefigs:
            plt.savefig(datetime_tag + '_graph_model_at_histogram_' + str(i))
        else:
            plt.show()

        plt.title(f'Liquid electron arrival time CDF')
        plot_at_hist(y_cumsum, label=f'True electron arrivals')
        plot_at_hist(y_hat_cumsum, label=f'Predicted electron arrivals')
        plt.legend()
        if savefigs:
            plt.savefig(datetime_tag + '_graph_model_at_cdf_' + str(i))
        else:
            plt.show()
    

def conv_graph_electron_arrival_prediction(XC, at_channel_hist, epochs=100, dim3=False, savefigs=False):
    # plot_hit_comparison(at_channel_hist[0], XC[0])
    AT = np.sum(at_channel_hist, axis=(1, 2))

    if dim3:
        graph_model = Conv3DGraphElectronModel(
            graph_layer_sizes   = [4, 4, 1], 
            layer_sizes         = [512, 700],
        )
    else:
        graph_model = ConvGraphElectronModel(
            graph_layer_sizes   = [100, 100, 100], 
            layer_sizes         = [512, 700],
        )

    # graph_model = load('conv_graph_model' + ('_3d' if dim3 else ''))
    if dim3:
        XC = np.reshape(XC, (XC.shape[0], 16, 16, 700, 1))
        graph_model.build((None, 16, 16, 700, 1))
    else:
        graph_model.build((None, 16, 16, 700))
    graph_model.compile(loss=MeanAbsoluteEMDLoss(), optimizer=tf.keras.optimizers.Adam(learning_rate=4e-4))
    graph_model.summary()

    if epochs > 0:
        train(graph_model, XC, AT, epochs=epochs, batch_size=128)#, plot_history=True)

    save_model(graph_model, 'conv_graph_model' + ('_3d' if dim3 else ''))

    x_test, y_test = XC[:5000], AT[:5000]
    y_hat_test = graph_model(x_test)
    y_hat_test = np.array(y_hat_test, dtype=np.float32)
    y_test_cumsum = np.cumsum(y_test, axis=1)
    y_hat_test_cumsum = np.cumsum(y_hat_test, axis=1)
    test_cumsum_error = np.abs(y_test_cumsum - y_hat_test_cumsum)
    cumsum_error_16_84 = np.percentile(test_cumsum_error, [16, 84], axis=0)
    mean_total_electrons = np.mean(y_test_cumsum, axis=0)

    plt.figure(figsize=(8, 6))
    plt.tight_layout()
    plt.fill_between(np.arange(700), cumsum_error_16_84[0], cumsum_error_16_84[1], alpha=0.5, label='16-84% error')
    plt.plot(np.mean(test_cumsum_error, axis=0), label='Mean error')
    # plt.plot(mean_total_electrons, label='Mean electron count')
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('Number of electrons')
    plt.title('Mean cumulative sum error for each sample')
    if savefigs:
        plt.savefig(datetime_tag + '_conv_graph_model_cumsum_error' + ('_3d' if dim3 else ''))
    else:
        plt.show()

    counts = np.sum(at_channel_hist, axis=(1, 2, 3))
    # Plot error by number of electrons in pulse
    errors_by_counts = np.zeros((counts.shape[0], np.max(counts) + 1))
    count_by_counts = np.zeros((counts.shape[0], np.max(counts) + 1))
    for i in tqdm(range(0, XC.shape[0], 5000)):
        x_test, y_test = XC[i:i+5000], AT[i:i+5000]
        y_hat_test = graph_model(x_test)
        y_hat_test = np.array(y_hat_test, dtype=np.float32)
        y_test_cumsum = np.cumsum(y_test, axis=1)
        y_hat_test_cumsum = np.cumsum(y_hat_test, axis=1)
        test_cumsum_error = np.abs(y_test_cumsum - y_hat_test_cumsum)
        for count, error in zip(counts, test_cumsum_error):
            errors_by_counts[count] += np.sum(error)
            count_by_counts[count] += 1
    errors_by_counts /= count_by_counts
    plt.figure(figsize=(8, 6))
    plt.tight_layout()
    plt.plot(errors_by_counts)
    plt.xlabel('Number of electrons in pulse')
    plt.ylabel('Mean absolute EMD error')
    plt.title('Mean error by number of electrons in pulse')
    if savefigs:
        plt.savefig(datetime_tag + '_conv_graph_model_error_by_count' + ('_3d' if dim3 else ''))
    else:
        plt.show()

    
    for i in range(10):
        at_i = np.array(AT[i], dtype=np.float32)
        at_hat_i = np.array(graph_model(np.expand_dims(XC[i], axis=0))[0], dtype=np.float32)

        y_cumsum = tf.cumsum(at_i, axis=0)
        y_hat_cumsum = tf.cumsum(at_hat_i, axis=0)

        sum_mse = tf.reduce_mean(tf.square(y_cumsum - y_hat_cumsum))

        dprint(f'EMD MSE: {sum_mse : .4f}')

        plt.title(f'Liquid electron arrival time histogram')
        plt.figure(figsize=(8, 6))
        plt.tight_layout()
        plot_at_hist(at_i, label=f'True electron arrivals')
        plot_at_hist(at_hat_i, label=f'Predicted electron arrivals')
        plt.legend()
        if savefigs:
            plt.savefig(datetime_tag + f'_conv_graph_model_{i}_hist' + ('_3d' if dim3 else ''))
        else:
            plt.show()

        plt.title(f'Liquid electron arrival time CDF')
        plt.figure(figsize=(8, 6))
        plt.tight_layout()
        plot_at_hist(y_cumsum, label=f'True electron arrivals')
        plot_at_hist(y_hat_cumsum, label=f'Predicted electron arrivals')
        plt.legend()
        if savefigs:
            plt.savefig(datetime_tag + f'_conv_graph_model_{i}_cdf' + ('_3d' if dim3 else ''))
        else:
            plt.show()
    


    


def mlp_electron_arrival_prediction(X, AT, epochs=100):
    model = MLPElectronModel(layer_sizes=[1000, 700])

    # model = load('mlp_at_model')
    model.build((None, 256, 700))
    model.compile(loss=MeanAbsoluteEMDLoss(), optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4))
    model.summary()

    if epochs > 0:
        train(model, X, AT, epochs=epochs, batch_size=128, plot_history=True)
    
    save_model(model, 'mlp_at_model')

    x_test, y_test = X[:5000], AT[:5000]
    y_hat_test = model(x_test)
    y_hat_test = np.array(y_hat_test, dtype=np.float32)
    test_error = np.abs(y_test - y_hat_test)
    error_16_84 = np.percentile(test_error, [16, 84], axis=0)
    y_test_cumsum = np.cumsum(y_test, axis=1)
    y_hat_test_cumsum = np.cumsum(y_hat_test, axis=1)
    test_cumsum_error = np.abs(y_test_cumsum - y_hat_test_cumsum)
    cumsum_error_16_84 = np.percentile(test_cumsum_error, [16, 84], axis=0)

    mean_total_electrons = np.mean(y_test_cumsum, axis=0)

    plt.fill_between(np.arange(700), cumsum_error_16_84[0], cumsum_error_16_84[1], alpha=0.5, label='16-84% error')
    plt.plot(np.mean(test_cumsum_error, axis=0), label='Mean error')
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('Number of electrons')
    plt.title('Mean cumulative sum error for each sample')
    plt.show()

    for i in range(10):
        at_i = np.array(AT[i], dtype=np.float32)
        at_hat_i = np.array(model(np.expand_dims(X[i], axis=0))[0], dtype=np.float32)

        y_cumsum = tf.cumsum(at_i, axis=0)
        y_hat_cumsum = tf.cumsum(at_hat_i, axis=0)

        sum_mse = tf.reduce_mean(tf.square(y_cumsum - y_hat_cumsum))

        dprint(f'EMD MSE: {sum_mse : .4f}')

        plt.title(f'Liquid electron arrival time histogram')
        plot_at_hist(at_i, label=f'True electron arrivals')
        plot_at_hist(at_hat_i, label=f'Predicted electron arrivals')
        plt.legend()
        plt.show()

        plt.title(f'Liquid electron arrival time CDF')
        plot_at_hist(y_cumsum, label=f'True electron arrivals')
        plot_at_hist(y_hat_cumsum, label=f'Predicted electron arrivals')
        plt.legend()
        plt.show()

    


def test_graph_network():
    graph_adjacency_matrix = create_grid_adjacency(16)
    graph_model = GraphElectronModel(
        adjacency_matrix    = graph_adjacency_matrix,
        graph_layer_sizes   = [512, 256, 128, 64, 32, 16, 8], 
        layer_sizes         = [512, 700],
        loss                = tf.keras.losses.MeanSquaredError()
    )

    X = np.zeros((1, 16, 16, 700))
    X[0][3][12] = np.ones(700) * 100
    X[0][10][9] = np.ones(700) * 30
    X = X.reshape(X.shape[0], -1, 700)

    iterations = graph_model.debug_call(X)

    for iteration in iterations:
        iteration = np.sum(iteration, axis=2)
        iteration = iteration.reshape((16, 16))
        plt.imshow(iteration)
        plt.show()
