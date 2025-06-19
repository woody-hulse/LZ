import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import sys
import copy
import networkx as nx
import itertools
import h5py

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
from experiments import *
from autoencoder_ import *
from pulse import *
from generator import *

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
DMS_CHANNEL_FILE_ = '../dSSdMS/dSS_2400417_gaussgass_700samplearea7000_areafrac0o5_1.0e+05events_random_centered_channel_withEAT.npz'

DMS_CHANNEL_FILE = '../dSSdMS/dSS_20241110_gaussgass_700samplearea7000_5.0e+04events_random_centered.npz'
# SS events with associated (binned) photon arrivals
# DSS_CHANNEL_PHOTON_FILE = '../dSSdMS/dSS_2400501_gaussgass_700samplearea7000_areafrac0o5_1.0e+04events_random_centered.npz'
# DSS_CHANNEL_PHOTON_FILE = '../dSSdMS/dSS_2400920_gaussgass_700samplearea7000_areafrac0o5_5.0e+04events_random_centered.npz' # SIMPLE
# DSS_CHANNEL_PHOTON_FILE = '../dSSdMS/dSS_2400917_gaussgass_700samplearea7000_areafrac0o5_5.0e+04events_random_centered.npz'
# DSS_CHANNEL_PHOTON_FILE = '../dSSdMS/dSS_2400921_gaussgass_700samplearea7000_areafrac0o5_5.0e+04events_random_centered.npz' # NORMAL
DSS_CHANNEL_PHOTON_FILE = '../dSSdMS/dSS_2400928_gaussgass_700samplearea7000_areafrac0o5_5.0e+04events_random_centered.npz' # RANDOM

DMS_RANDOM_FILE = '../dSSdMS/dMS_20241028_gaussgass_700samplearea7000_areafrac0o5_1.0e+05events_random_centered.npz'
DSS_FILE = '../dSSdMS/dSS_20241117_gaussgass_700samplearea7000_1.0e+04events_random_centered.npz'

class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, ckpt_manager):
        super().__init__()
        self.ckpt_manager = ckpt_manager

    def on_epoch_end(self, epoch, logs=None):
        save_path = self.ckpt_manager.save()
        print(f'\nsaved checkpoint for epoch {epoch + 1}: {save_path}')

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

    # X, XC, PXC, Y, AT = load_pulse_dataset(DMS_CHANNEL_FILE)
    # X, Y = load_pulse_dataset_old(DMS_FILE)
    X, XC, C, PXC, EXC = load_SS_dataset(DSS_FILE)
    # X_continuous, Y_continuous = load_pulse_dataset_old(DMS_CONTINUOUS_FILE)
    # X_ms, MU = load_random_ms_pulse_dataset(DMS_RANDOM_FILE)
    # X, XC, PXC, Y, AT = load_pulse_dataset(DMS_CHANNEL_FILE_)

    '''
    for i in range(len(MU)):
        MU[i] = np.sort(MU[i])

    X_ms_binned = [[] for i in range(20)]
    for i in tqdm(range(len(X_ms))):
        dmu = MU[i][1] - MU[i][0]
        X_ms_binned[int(dmu) // 5].append(X_ms[i])
    min_bin_size = min([len(X_ms_binned[i]) for i in range(20)])
    X_ms_binned = np.array([X_ms_binned[i][:min_bin_size] for i in range(20)])
    '''

    # X_binned = np.array([X[i * 5000 : (i + 1) * 5000] for i in range(20)])
    # Y_binned = np.array([Y[i * 5000 : (i + 1) * 5000] for i in range(20)])

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # X_XC_PXC_Y_AT_train, X_XC_PXC_Y_AT_test = train_test_split_all([X, XC, PXC, Y, AT], test_size=0.2)
    
    event_indices = np.array([i for i in range(min(X.shape[0], num_events))])
    np.random.shuffle(event_indices)
    np.random.shuffle(event_indices)
    X       = X[event_indices][:num_events]     # Summed pulse signal
    XC      = XC[event_indices][:num_events]    # Channel pulse signal
    PXC     = PXC[event_indices][:num_events]   # Photon count channel pulse signal
    EXC     = EXC[event_indices][:num_events]   # Electron count channel pulse signal
    Y       = Y[event_indices][:num_events]     # Delta mu for MS, 0 for SS
    AT      = AT[event_indices][:num_events]    # Electron arrival times
    AF      = AF[event_indices][:num_events]    # Area fraction of MS pulse
    C       = C[event_indices][:num_events]     # XY of SS pulse

    # X_ms    = X_ms[event_indices][:num_events]  # MS pulse signal
    # MU      = MU[event_indices][:num_events]    # Means of MS pulses

    # X_continuous = X_continuous[event_indices][:num_events]
    # Y_continuous = Y_continuous[event_indices][:num_events]

    # num_samples = 1000000
    # sample_indices = np.arange(num_samples)
    # np.random.shuffle(sample_indices)
    # X_FLAT = np.reshape(X, (-1, 1))
    # XC_FLAT = np.reshape(XC, (-1, 1))
    # PXC_FLAT = np.reshape(PXC, (-1, 1))
    # PX_FLAT = np.reshape(np.sum(PXC, axis=(1, 2)), (-1, 1))
    '''
    Experiments
    '''
    
    
    
    '''
    
    D = 4
    N = 4

    # ones = np.array(np.random.uniform(0, 1, (2, N * N * 10 + N)), dtype=np.float32)
    # y_ones = np.array(np.random.uniform(0, 1, (2, N, 4)), dtype=np.float32)
    # print(GraphMultivariateNormalModel.pdf_loss(y_ones, ones))

    # X, XC, Y, C, P, E = generate_N_channel_scatter_events(X, XC, C, PXC, EXC, max_N=N, num_events=int(1e4))
    # np.savez_compressed('../dSSdMS/ms_data1e4.npz', X=X, XC=XC, Y=Y, C=C, P=P, E=E)
    # X, XC, Y, C, P, E = np.load('../dSSdMS/ms_data.npz').values()
    adjacency_matrix = create_grid_adjacency(XC.shape[1])
    cholesky_mask = GraphMVDModel.make_cholesky_mask(num_dims=D, correlated_dims=[[0, 1]])
    
    train_split = int(0.9 * X.shape[0])
    
    batch_size=256
    data_generator = N_channel_scatter_events_generator(XC[:train_split], C[:train_split], PXC[:train_split], max_N=N, batch_size=batch_size)
    validation_data_generator = N_channel_scatter_events_generator(XC[train_split:], C[train_split:], PXC[train_split:], max_N=N, batch_size=batch_size)

    model = GraphMVDModel(adjacency_matrix=adjacency_matrix, num_dists=N, num_dims=D)# , cholesky_mask=cholesky_mask)
    model.build((None, XC.shape[1], XC.shape[2]))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer, 
        loss=GraphMVDModel.get_partial_func(func=GraphMVDModel.combined_loss, num_dists=N, num_dims=D),
        metrics=[
            GraphMVDModel.get_partial_func(func=GraphMVDModel.pdf_loss, num_dists=N, num_dims=D),
            GraphMVDModel.get_partial_func(func=GraphMVDModel.mask_loss, num_dists=N, num_dims=D),
        ]
    )
    
    dummy_batch = next(iter(data_generator))
    model.train_on_batch(dummy_batch[0], dummy_batch[1])

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=f'saved_{model.name}_ckpt', max_to_keep=3)
    
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).assert_existing_objects_matched()
        print('restored from checkpoint:', ckpt_manager.latest_checkpoint)
    else:
        print('no checkpoint found--initializing from scratch.')

    checkpoint_callback = CheckpointCallback(ckpt_manager)
    
    history = model.fit(
        data_generator,
        epochs=10,
        verbose=1,
        validation_data=validation_data_generator,
        steps_per_epoch=XC.shape[0] // batch_size,
        validation_steps=4,
        callbacks=[checkpoint_callback]
    )
    
    X, XC, Y, C, P, E = generate_N_channel_scatter_events(X, XC, C, PXC, EXC, max_N=N, num_events=int(1e2))
    plot_graph_mvd_examples(XC, Y, C, P, model, pdf_func=normal_pdf, num_examples=10, N=N)
    
    '''
    
    '''
    
    LOAD = 1
    TRAIN = 1
    
    custom_objects = {
        'combined_loss': GraphMultivariateNormalModel.combined_loss,
        'pdf_loss': GraphMultivariateNormalModel.pdf_loss,
        'mask_loss': GraphMultivariateNormalModel.mask_loss
    }
    
    if LOAD:
        model = load_model('saved_models/graph_model__0.keras', custom_objects=custom_objects, compile=True)
    else:
        model = GraphMultivariateNormalModel(adjacency_matrix=adjacency_matrix)
        
        model.build((None, XC.shape[1], XC.shape[2]))
        
        model.compile(
            optimizer=Adam(learning_rate=1e-3), 
            loss=GraphMultivariateNormalModel.combined_loss, 
            metrics=[
                GraphMultivariateNormalModel.pdf_loss,
                GraphMultivariateNormalModel.mask_loss,
            ]
        )
    

    # XYZP = np.concatenate([C, Y, P], axis=-1)
    if TRAIN:
        for i in range(1):
            model.fit(data_generator, epochs=1, verbose=1, validation_data=validation_data_generator, steps_per_epoch=XC.shape[0] // batch_size // 10, validation_steps=4)
            model.save(f'saved_models/graph_model__{i}.keras')
        # train(model, XC, XYZP, epochs=25, batch_size=512, validation_split=0.2, summary=True, plot_history=False)
        # for i in range(8):
        #     train(model, XC, XYZP, epochs=25, batch_size=128, validation_split=0.05, summary=True, plot_history=False)
        #     save_model(model, f'graph_multi_normal_model3_{i}')

    # GraphMultivariateNormalModel.decompose_output(model(XC[0:1]), show=True)
    '''
    
    return
    
    X, XC, Y, C, P, E = generate_N_channel_scatter_events(X, XC, C, PXC, EXC, max_N=N, num_events=int(1e2), normalize=True)
    XYZP = np.concatenate([C, Y, P], axis=-1)
    
    # plot_prediction_xyz_z_distribution(XC, XYZP, model, pdf_func=normal_pdf)
    # plot_pdf_heatmap_3d(XC, Y, C, P, model, pdf_func=GraphMultivariateNormalModel.multivariate_norm_xycorr_pdf, res=50, num_examples=10, N=N)
    plot_graph_multivariate_normal_examples(XC, Y, C, P, model, pdf_func=normal_pdf, num_examples=20, N=N)
    
    return
    '''

    N = 4
    D = 3

    # X, XC, Y, C, P, E = generate_N_channel_scatter_events(X, XC, C, PXC, EXC, max_N=N, num_events=int(1e3))
    # Y /= 700
    # C /= XC.shape[1]
    # P /= 10000
    # E /= 250

    # np.savez_compressed('../dSSdMS/ms_data_normalized_1e3_.npz', X=X, XC=XC, Y=Y, C=C, P=P, E=E)
    # X, XC, PXC, Y, C, P, E = np.load('../dSSdMS/ms_data_normalized_5e4.npz').values()
    # X, XC, Y, C, P, E = np.load('../dSSdMS/ms_data_normalized_1e4.npz').values()
    adjacency_matrix = create_grid_adjacency(XC.shape[1])
    
    batch_size = 256
    data_generator = N_channel_scatter_events_generator(XC, C, PXC, max_N=N, batch_size=batch_size)
    validation_data_generator = N_channel_scatter_events_generator(XC, C, PXC, max_N=N, batch_size=batch_size)
    
    # XC += np.random.normal(0, 0.1, (XC.shape[0], XC.shape[1], XC.shape[2], XC.shape[3]))
    
    # print(np.min(XC), np.max(XC))

    LOAD = 0
    TRAIN = 1
    
    if LOAD:
        model = load('graph_mvd_model_2')
    else:
        model = GraphMVDModel(num_dims=D, num_dists=N, adjacency_matrix=adjacency_matrix)
        model.build((None, XC.shape[1] * XC.shape[2], XC.shape[3]))
        
    model.compile(
        optimizer=Adam(learning_rate=2e-4), 
        loss=GraphMVDModel.get_partial_func(func=GraphMVDModel.combined_loss, num_dists=N, num_dims=D),
        metrics=[
            GraphMVDModel.get_partial_func(func=GraphMVDModel.pdf_loss, num_dists=N, num_dims=D),
            GraphMVDModel.get_partial_func(func=GraphMVDModel.mask_loss, num_dists=N, num_dims=D),
        ]
    )
    
    # ones = np.array(np.random.uniform(0, 1, (2, N * N * 10 + N)), dtype=np.float32)
    # y_ones = np.array(np.random.uniform(0, 1, (2, N, N)), dtype=np.float32)
    # print(GraphMVDModel.pdf_loss(y_ones, ones, N, D))
    
    if TRAIN:
        for i in range(10):
            model.fit(data_generator, epochs=5, verbose=1, steps_per_epoch=XC.shape[0] // batch_size)
            save_model(model, f'graph_mvd_model{i}')
        # for i in range(10):
        #     train(model, XC, XYZ, epochs=10, batch_size=256, validation_split=0.05, summary=True, plot_history=False)
        #     save_model(model, f'graph_mvd_model_{i}')
        # train(model, XC, XYZ, epochs=10, batch_size=256, validation_split=0.1, summary=True, plot_history=False)

    X, XC, Y, C, P, E = generate_N_channel_scatter_events(X, XC, C, PXC, EXC, max_N=N, num_events=int(1e2))
    XC = XC.reshape((XC.shape[0], XC.shape[1] * XC.shape[2], XC.shape[3]))
    XYZP = np.concatenate([C, Y, P], axis=-1)
    
    plot_mvn_pdf_heatmap_3d(XC, Y, C, P, model, pdf_func=GraphMVDModel.multivariate_normal_pdf, res=50, num_examples=10, N=N)
    plot_graph_nvm_examples(XC, Y, C, P, model, pdf_func=normal_pdf, num_examples=20, N=N)
    
    '''

    '''
    N = GraphDistributionModel.N
    # ones = np.ones((100, N * N * 7 + N))
    # y_ones = np.ones((100, N, 4))
    # print(GraphDistributionModel.combined_loss(y_ones, ones))

    C /= XC.shape[1]
    X, XC, Y, C, P, E = generate_N_channel_scatter_events(X, XC, C, PXC, EXC, max_N=N, num_events=int(1e3))
    adjacency_matrix = create_grid_adjacency(XC.shape[1])

    LOAD = 1
    TRAIN = 0

    print(tf.squeeze(tf.cast(Y[:10] != 0, tf.float32)))

    XC = XC.reshape((XC.shape[0], XC.shape[1] * XC.shape[2], XC.shape[3]))
    if LOAD:
        model = load('dist_graph_model_count') # dist_graph_model3
    else:
        model = GraphDistributionModel(adjacency_matrix=adjacency_matrix)
        model.build((None, XC.shape[1], XC.shape[2]))
    model.compile(
        optimizer=Adam(learning_rate=1e-3), 
        loss=GraphDistributionModel.combined_loss, 
        metrics=[
            GraphDistributionModel.z_loss,
            # GraphDistributionModel.xy_loss,
            GraphDistributionModel.xy_sum_loss,
            GraphDistributionModel.mask_loss,
            # GraphDistributionModel.all_loss,
            # GraphDistributionModel.count_loss_simple
    ])

    Y_ = np.concatenate([Y, C, P], axis=-1)
    print(Y_.shape, Y.shape, C.shape, P.shape)
    if TRAIN:
        train(model, XC, Y_, epochs=100, batch_size=512, validation_split=0.2, summary=True, plot_history=False)
        save_model(model, 'dist_graph_model_count')    

    # predictions = model.predict(XC)
    # z_preds = predictions[:, N:N + 2]
    # dmu_preds = np.abs(z_preds[:, 0] - z_preds[:, 1]) * 1000
    # print(dmu_preds[:40], Y[:40])

    output = model.predict(XC[:8])

    z_means = output[:, :N * N]
    z_stds  = output[:, N * N:2 * N * N]
    x_means = output[:, 2 * N * N:3 * N * N]
    x_stds  = output[:, 3 * N * N:4 * N * N]
    y_means = output[:, 4 * N * N:5 * N * N]
    y_stds  = output[:, 5 * N * N:6 * N * N]
    counts  = output[:, 6 * N * N:7 * N * N]

    # print(C[7])
    # print('.', x_means[7], x_stds[7], y_means[7], y_stds[7])

    print(Y_[7, :, 3])
    print('.', counts[7])

    # linearity_plot((X, DMU), dmu_preds, delta_mu=50, num_delta_mu=20, num_samples=1000, title='NDistSpatial ')
    # plot_prediction_xyz_z_distribution(XC, Y_, model, pdf_func=normal_pdf)
    plot_n_distribution_spatial_model_combined_examples(XC, Y, C, P, model, pdf_func=normal_pdf, num_examples=20, N=N)

    

    return

    print(AT.shape, X.shape)

    AT_sum = np.sum(AT, axis=(1, 2))

    model1 = MLPModel()
    model1.build((None, 700))
    model1.compile(optimizer=Adam(learning_rate=1e-4), loss=MeanAbsoluteError())

    model2 = MLPModel()
    model2.build((None, 700))
    model2.compile(optimizer=Adam(learning_rate=1e-4), loss=MeanAbsoluteError())

    train(model1, AT_sum, Y, epochs=30, batch_size=512, validation_split=0.2, summary=True, plot_history=False)

    train(model2, X, Y, epochs=30, batch_size=512, validation_split=0.2, summary=True, plot_history=False)
    '''

    return
    
    '''
    adjacency_matrix = create_grid_adjacency(XC.shape[1])
    fig = plt.figure(dpi=200)
    plt.imshow(adjacency_matrix, cmap='Greys', interpolation='none')

    G = nx.from_numpy_array(adjacency_matrix)

    # Plot the graph
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=False, node_size=50, node_color='blue', edge_color='black', width=0.5)
    plt.title('GNN adjacency matrix')
    # Display the graph
    plt.show()
    '''

    max_N = 8
    assert MLPNDistributionModel.N == max_N
    X, Y = generate_N_scatter_events(X, max_N=max_N, num_events=100000)

    # model = MLPNDistributionModel()
    # model.build((None, 700))
    # model.compile(optimizer=Adam(learning_rate=4e-4), loss=MLPNDistributionModel.combined_loss, metrics=[MLPNDistributionModel.pdf_loss, MLPNDistributionModel.mask_loss])
    model = load('n_dist_model8')
    model.compile(optimizer=Adam(learning_rate=4e-4), loss=MLPNDistributionModel.combined_loss, metrics=[MLPNDistributionModel.pdf_loss, MLPNDistributionModel.mask_loss])
    
    # test_y = np.random.randint(0, 2, (20, max_N)) 
    # test_y_hat = np.ones((20, max_N * max_N * 2 + max_N))
    # print(MLPNDistributionModel.combined_loss(Y[:20], test_y_hat))

    train(model, X, Y, epochs=20, batch_size=512, validation_split=0.2, summary=True, plot_history=False)

    save_model(model, 'n_dist_model8')

    plot_n_distribution_model_combined_examples(X, Y, model, pdf_func=normal_pdf, num_examples=10, N=max_N)

    # plot_n_distribution_model_examples(X, Y, model, pdf_func=normal_pdf, num_examples=10)
    


    # print(np.sum(XC[0]), np.sum(AT[0]))
    # print(np.sum(np.abs(XC[
    # 0] - AT[0])), np.sum(XC[0] - AT[0]))
    # plot_hit_pattern(XC[0])
    # graph_electron_arrival_prediction(XC, AT[:, :, -1], epochs=50)
    # graph_channel_electron_arrival_prediction(XC, AT, epochs=0)
    # graph_electron_arrival_prediction(np.array(PXC, np.float32), AT, epochs=30)
    # PXC_sum = np.sum(PXC, axis=(1, 2))
    # XC_sum = np.sum(XC, axis=(1, 2))
    
    # XC_n = get_grid_neighborhood(XC, size=3)
    # print(XC_n.shape)

    # graph_electron_arrival_prediction(XC, AT, epochs=10)
    # conv_graph_electron_arrival_prediction(XC, AT, epochs=25, dim3=False, savefigs=True)
    # conv_graph_electron_arrival_prediction(XC, AT, epochs=6, dim3=True, savefigs=True)

    # model = MLPModel()
    # model.compile(optimizer=Adam(learning_rate=1e-4), loss=mean_log_error, metrics=[MeanAbsoluteError()])
    # model.build((None, 700))
    # train(model, X, Y, epochs=100, batch_size=512, validation_split=0.2, summary=True, plot_history=False)
    # model.fit(X, Y, epochs=100, batch_size=512, validation_split=0.2, verbose=1)

    # plot_histogram(X_binned, Y_binned, model)

    # maf = MAFDensityEstimator(700, 3, 128)
    # train_maf(maf, X_continuous, Y_continuous)

    '''
    for i in range(3):
        plt.plot(X_ms[i])
        plt.show()

    
    mu_model = MLPDoubleNormalDistributionModel()
    mu_model.build((None, 700))
    mu_model = load('double_normal_dist_model')
    mu_model.compile(optimizer=Adam(learning_rate=2e-6), loss=MLPDoubleNormalDistributionModel.loss, metrics=[MLPDoubleNormalDistributionModel.dmu_loss])
    # train(mu_model, X_ms, MU - 350, epochs=2, batch_size=512, validation_split=0.2, summary=True, plot_history=True)

    print(MU[:40, 1] - MU[:40, 0])
    print((mu_model.predict(X_ms[:40])[:, 2] - mu_model.predict(X_ms[:40])[:, 0]) * 1000)
    
    
    # save_model(mu_model, 'double_normal_dist_model')

    plot_double_distribution_model_examples(X_ms, MU - 350, mu_model, pdf_func=normal_pdf, num_examples=10, split=2)

    # linearity_plot_2(X_ms_binned, mu_model, output_func=lambda x: np.abs(x[:, 2] - x[:, 0]) * 10000)
    '''

    '''
    conf_model = load('skew_dist_conf_model')

    # conf_model = MLPDistributionModel()
    # conf_model.build((None, 700))
    conf_model.compile(optimizer=Adam(learning_rate=3e-6), loss=skewnormal_abs_pdf_loss, metrics=[mu_loss, skew_mu_loss])

    train(conf_model, X_continuous, Y_continuous, epochs=50, batch_size=512, validation_split=0.2, summary=True, plot_history=True)
    save_model(conf_model, 'abs_dist_conf_model')

    print(conf_model(X_continuous[:40]))

    linearity_plot_2(X_binned, conf_model, output_func=lambda x: np.abs(abs_skew_mu(x) * 1000))
    '''

    '''
    conf_model = MLPDistributionModel(predict_ss=True)
    SS = np.where(Y == 0, 1, 0)

    conf_model = load('ss_dist_conf_model')
    # conf_model.build((None, 700))
    # conf_model.compile(optimizer=Adam(learning_rate=1e-5), loss=MLPDistributionModel.ss_loss)
    # train(conf_model, X, np.vstack([Y, SS]).T, epochs=100, batch_size=512, validation_split=0.2, summary=True, plot_history=False)

    # save_model(conf_model, 'ss_dist_conf_model')

    # mlp = load('mlp_model')
    mlp = MLPModel()
    mlp.build((None, 700))
    mlp.compile(optimizer=Adam(learning_rate=1e-4), loss=BinaryCrossentropy())
    train(mlp, X, SS, epochs=20, batch_size=512, validation_split=0.2, summary=True, plot_history=False)
    # save_model(mlp, 'mlp_model')

    
    roc(X, SS, [mlp, conf_model])
    # plot_mae_vs_dmu(X_binned, Y_binned, [mlp, conf_model], mu_funcs=[lambda x: x, abs_skew_mu])
    '''




    # plot_prediction_percentile_distribution_normal(X_continuous, Y_continuous, conf_model, skewnormal_abs_pdf)

    # plot_prediction_z_distribution_normal(X, Y, conf_model, normal_pdf)

    # plot_distribution_histogram(X_binned, Y_binned, conf_model, pdf_func=skewnormal_abs_pdf)
    # plot_distribution_model_examples(X, Y, conf_model, pdf_func=skewnormal_abs_pdf, num_examples=20)
    # plot_pdf_gradient_histogram(X_binned, Y_binned, conf_model)
    

    # conv_graph_electron_arrival_prediction(XC[:10000], AT[:10000], epochs=10, dim3=True, savefigs=True)
    
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

    '''

    mlp_model = MLPModel()
    mlp_model.build((None, 700))
    mlp_model = load('mlp_model')
    mlp_model.compile(optimizer=Adam(learning_rate=1e-4), loss=MeanAbsoluteError())
    # train(mlp_model, X_continuous, Y_continuous, epochs=40, batch_size=512, validation_split=0.2, summary=True, plot_history=False)
    # save_model(mlp_model, 'mlp_model')

    custom_conf_model = MLPCustomDistributionModel()
    custom_conf_model = load('custom_conf_model')
    custom_conf_model.compile(optimizer=Adam(learning_rate=1e-4), loss=distribution_loss, metrics=[tf.keras.losses.SparseCategoricalCrossentropy(), eval_distribution_loss])

    trials = 100
    mlp_convergence_losses = np.zeros(trials)
    custom_convergence_losses = np.zeros(trials)

    mlp_convergence_validation_losses = np.zeros(trials)
    custom_convergence_validation_losses = np.zeros(trials)
    for i in range(trials):
        history_mlp = train(mlp_model, X_continuous, Y_continuous, epochs=1, batch_size=512, validation_split=0.2, summary=False, plot_history=False)
        history_custom = train(custom_conf_model, X_continuous, np.array(Y_continuous, dtype=np.int32), epochs=1, batch_size=512, validation_split=0.2, summary=False, plot_history=False)
        mlp_convergence_losses[i] = history_mlp.history['loss'][-1]
        custom_convergence_losses[i] = history_custom.history['eval_distribution_loss'][-1]
        mlp_convergence_validation_losses[i] = history_mlp.history['val_loss'][-1]
        custom_convergence_validation_losses[i] = history_custom.history['val_eval_distribution_loss'][-1]
    
    df_losses = pd.DataFrame({
        'Model': ['MLP Model'] * trials * 2 + ['Custom Conf Model'] * trials * 2,
        'Type': ['Train'] * trials + ['Test'] * trials + ['Train'] * trials + ['Test'] * trials,
        'MAE': np.concatenate([
            mlp_convergence_losses, mlp_convergence_validation_losses,
            custom_convergence_losses, custom_convergence_validation_losses
        ])
    })

    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df_losses, x='Model', y='MAE', hue='Type', split=True)
    plt.title('MLP vs Distribution model MAE after convergence: 100 trials')
    plt.show()

    '''
    
    '''
    custom_conf_model = MLPCustomDistributionModel()

    custom_conf_model = load('custom_conf_model_2nd')

    custom_conf_model.compile(optimizer=Adam(learning_rate=1e-4), loss=distribution_loss, metrics=[tf.keras.losses.SparseCategoricalCrossentropy(), eval_distribution_loss])
    train(custom_conf_model, X_continuous, np.array(Y_continuous, dtype=np.int32), epochs=20, batch_size=512, validation_split=0.2, summary=True, plot_history=False)

    save_model(custom_conf_model, 'custom_conf_model_2nd')

    plot_prediction_percentile_distribution_normal(X_continuous, Y_continuous, custom_conf_model, identity_pdf)
    plot_distribution_histogram(X_binned, Y_binned, custom_conf_model, pdf_func=identity_pdf)
    plot_distribution_model_examples(X, Y, custom_conf_model, pdf_func=identity_pdf, num_examples=20)
    '''

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