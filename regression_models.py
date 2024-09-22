import tensorflow as tf
import spektral
# import keras_tuner
import numpy as np
from vgg import *

from scipy.stats import wasserstein_distance
from matplotlib import pyplot as plt


# VGG16-based 1D convolutional neural network
class ConvModel(tf.keras.Model):
    def __init__(self, input_size=None, output_size=1, name='conv_1d_model'):
        super().__init__(name=name)

        # hyperparameters
        num_block1 = 1
        num_block2 = 2
        num_block3 = 3
        num_block4 = 4
        num_block5 = 5
        num_dense = 2

        block1 = [tf.keras.layers.Convolution1D(filters=32, kernel_size=3, strides=1) for _ in range(num_block1)]
        block2 = [tf.keras.layers.Convolution1D(filters=32, kernel_size=3, strides=1) for _ in range(num_block2)]
        block3 = [tf.keras.layers.Convolution1D(filters=64, kernel_size=5, strides=1) for _ in range(num_block3)]
        block4 = [tf.keras.layers.Convolution1D(filters=64, kernel_size=5, strides=1) for _ in range(num_block4)]
        block5 = [tf.keras.layers.Convolution1D(filters=64, kernel_size=7, strides=1) for _ in range(num_block5)]

        self.blocks = [block1, block2, block3, block4, block5]
        self.pooling_layer = tf.keras.layers.MaxPooling1D(pool_size=2)
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layers = [tf.keras.layers.Dense(128, activation='relu') for _ in range(num_dense)]
        self.output_layer = tf.keras.layers.Dense(1)

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.MeanAbsoluteError()

        if input_size:
            self.compile(optimizer=self.optimizer, loss=self.loss)
            self.build((None, input_size, 1))
    
    def call(self, x):
        for block in self.blocks:
            for layer in block:
                x = layer(x)
            x = self.pooling_layer(x)
        
        x = self.flatten_layer(x)
        for layer in self.dense_layers:
            x = layer(x)
        
        x = self.output_layer(x)

        return x
    

class ConvChannelModel(tf.keras.Model):
    def __init__(self, input_size=None, kernel_size=3, num_filters=128, num_conv=4, layer_sizes=[1], classification=False, name='conv_channel_model_'):
        for size in layer_sizes:
            name += str(size) + '-'
        name = name[:-1]
        super().__init__(name=name)
        
        self.conv_layers = [tf.keras.layers.Conv2D(num_filters, kernel_size, activation='leaky_relu', padding='valid') for _ in range(num_conv)]
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layers = [tf.keras.layers.Dense(size, activation='leaky_relu') for size in layer_sizes[:-1]]

        metrics = []
        self.optimizer = tf.keras.optimizers.Adam(1e-5)
        if classification: 
            self.dense_layers.append(tf.keras.layers.Dense(layer_sizes[-1], activation='sigmoid'))
            self.loss = tf.keras.losses.BinaryCrossentropy()
            metrics.append('AUC')
        else: 
            self.dense_layers.append(tf.keras.layers.Dense(layer_sizes[-1], activation='linear'))
            self.loss = tf.keras.losses.MeanSquaredError()

        if input_size:
            self.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics)
            self.build([None] + input_size)

    def call(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten_layer(x)
        for layer in self.dense_layers:
            x = layer(x)
        
        return x
    

class MLPChannelModel(tf.keras.Model):
    def __init__(self, input_size=None, head_sizes=[1], layer_sizes=[1], heads=3, classification=False, name='mlp_model_'):
        for size in layer_sizes:
            name += str(size) + '-'
        name = name[:-1]
        super().__init__(name=name)
        
        self.heads = heads
        self.reshape_layer = tf.keras.layers.Reshape((-1, input_size))
        self.td_heads = [tf.keras.Sequential([tf.keras.layers.Dense(size, activation='leaky_relu') for size in head_sizes]) for _ in range(heads)]
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layers = [
            tf.keras.layers.Dense(512, activation='leaky_relu'),
            tf.keras.layers.Dense(256, activation='leaky_relu')
        ]

        metrics = []
        self.optimizer = tf.keras.optimizers.Adam()
        if classification: 
            self.dense_layers.append(tf.keras.layers.Dense(layer_sizes[-1], activation='sigmoid'))
            self.loss = tf.keras.losses.BinaryCrossentropy()
            metrics.append('AUC')
        else: 
            self.dense_layers.append(tf.keras.layers.Dense(layer_sizes[-1], activation='linear'))
            # self.loss = tf.keras.losses.MeanSquaredError()
            self.loss = tf.keras.losses.MeanSquaredError()

        if input_size:
            self.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics)
            self.build((None, 10, 10, input_size))

    def call(self, x):
        x = self.reshape_layer(x)
        head_out = []
        for head in self.td_heads:
            head_out.append(head(x))
        x = tf.concat(head_out, axis=1)
        x = self.flatten_layer(x)
        for layer in self.dense_layers:
            x = layer(x)
        
        return x


class DummyModel(tf.keras.Model):
    def __init__(self, name='dummy_model'):
        super().__init__(name=name)
    
    def call(self, x):
        return tf.clip_by_value(x, 0, 1)


class ChannelScaledMeanSquaredError(tf.keras.losses.Loss):
    def __init__(self, delta=2.0):
        super().__init__()
        self.delta = tf.cast(delta, tf.float32)
    
    def call(self, y, y_hat):
        scale = tf.pow(self.delta, tf.cast(y, tf.float32))
        squared_error = tf.square(y - y_hat)
        scaled_mean_square_error = tf.reduce_mean(scale * squared_error)

        y_sum = tf.reduce_sum(y, axis=(1, 2))
        y_hat_sum = tf.reduce_sum(y_hat, axis=(1, 2))
        mean_squared_error = tf.reduce_sum(tf.square(y_sum - y_hat_sum))

        return scaled_mean_square_error + mean_squared_error


class ScaledMeanSquaredError(tf.keras.losses.Loss):
    def __init__(self, delta=2.0):
        super().__init__()
        self.delta = tf.cast(delta, tf.float32)
    
    def call(self, y, y_hat):
        scale = tf.pow(self.delta, tf.cast(y, tf.float32))
        squared_error = tf.pow(tf.abs(y - y_hat), self.delta)
        scaled_mean_square_error = tf.reduce_mean(scale * squared_error)
        return scaled_mean_square_error


class BaselinePhotonModel(tf.keras.Model):
    def __init__(self, input_size=None, layer_sizes=[1], max=4, delta=1.6, name='baseline_photon_model_'):
        for size in layer_sizes:
            name += str(size) + '-'
        name = name[:-1]
        super().__init__(name=name)

        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layers = [tf.keras.layers.Dense(size, activation='sigmoid') for size in layer_sizes[:-1]]
        self.dense_layers.append(tf.keras.layers.Dense(layer_sizes[-1], activation='sigmoid'))

        self.rescale_layer = tf.keras.layers.Rescaling(max + 1, offset=-1e-2)
        self.optimizer = tf.keras.optimizers.Adam()
        # self.loss = tf.keras.losses.MeanSquaredError()
        self.loss = ScaledMeanSquaredError(delta=delta)
        if input_size:
            self.compile(optimizer=self.optimizer, loss=self.loss)
            self.build((None, input_size,))
    
    def call(self, x):
        x = self.flatten_layer(x)
        for layer in self.dense_layers:
            x = layer(x)
        x = self.rescale_layer(x)
        return x
    
    def build(self, input_shape):
        super().build(input_shape)
        self.rescale_layer.build(input_shape)
        self.built = True


class MeanSquaredWassersteinLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
    
    def call(self, y, y_hat):
        y_cumsum = tf.cumsum(y, axis=2)
        y_hat_cumsum = tf.cumsum(y_hat, axis=2)
        return tf.math.reduce_mean((y_cumsum - y_hat_cumsum) ** 2)

class MeanSquaredEMDLoss3D(tf.keras.losses.Loss):
    def __init__(self, rows=16, cols=16, features=700):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.features = features
    
    def call(self, y, y_hat):
        y       = tf.reshape(y,     (-1, self.rows, self.cols, self.features))
        y_hat   = tf.reshape(y_hat, (-1, self.rows, self.cols, self.features))

        y_cumsum = tf.cumsum(y, axis=3)
        y_cumsum_x = tf.cumsum(y_cumsum, axis=1)
        y_cumsum_xy = tf.cumsum(y_cumsum_x, axis=2)

        y_hat_cumsum = tf.cumsum(y_hat, axis=3)
        y_hat_cumsum_x = tf.cumsum(y_hat_cumsum, axis=1)
        y_hat_cumsum_xy = tf.cumsum(y_hat_cumsum_x, axis=2)

        y_cumsum_sum = tf.reduce_sum(y_cumsum, axis=[1, 2])
        y_hat_cumsum_sum = tf.reduce_sum(y_hat_cumsum, axis=[1, 2])

        channel_mse = tf.math.reduce_mean(tf.square(y_cumsum_xy - y_hat_cumsum_xy))
        sum_mse = tf.math.reduce_mean(tf.square(y_cumsum_sum - y_hat_cumsum_sum))

        return channel_mse + sum_mse

    def get_config(self):
        return {
            'rows': self.rows,
            'cols': self.cols,
            'features': self.features
        }


class MeanSquaredEMDLoss(tf.keras.losses.Loss):
    def __init__(self, rows=16, cols=16, features=700):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.features = features
    
    def call(self, y, y_hat):
        y_cumsum = tf.cumsum(y, axis=1)
        y_hat_cumsum = tf.cumsum(y_hat, axis=1)

        return tf.math.reduce_mean((y_cumsum - y_hat_cumsum) ** 2)

    def get_config(self):
        return {
            'rows': self.rows,
            'cols': self.cols,
            'features': self.features
        }


class ElectronProbabilityLoss(tf.keras.losses.Loss):
    DIFFWIDTH       = 300   # ns
    G2              = 47.35
    EGASWIDTH       = 450   # ns
    PHDWIDTH        = 20    # ns
    SAMPLERATE      = 10    # ns
    NUM_ELECTRONS   = 148

    def __init__(self):
        super().__init__()
    
    def call(self, generated_electron_hits, photon_hits):
        return np.random.random(size=generated_electron_hits.shape[0])


class MLPElectronModel(tf.keras.Model):
    def __init__(self, layer_sizes=[700], name='mlp_electron_model_', *args, **kwargs):
        super().__init__(name=name, **kwargs)
        
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1e-6)
        self.dense_layers = [tf.keras.layers.Dense(size, activation='relu', kernel_initializer=initializer) 
                             for size in layer_sizes[:-1]]
        self.dense_layers.append(tf.keras.layers.Dense(layer_sizes[-1], activation='sigmoid', kernel_initializer=initializer))
        self.rescale_layer = tf.keras.layers.Rescaling(1, offset=-1e-4)
    
    def call(self, x):
        for layer in self.dense_layers:
            x = tf.keras.layers.TimeDistributed(layer)(x)
        x = self.rescale_layer(x)
        return x


class GraphElectronModel(tf.keras.Model):
    def __init__(self, adjacency_matrix, graph_layer_sizes=[1], layer_sizes=None, loss=None, **kwargs):
        super(GraphElectronModel, self).__init__(**kwargs)
        
        if type(adjacency_matrix) == dict: # for deserialization
            adjacency_matrix = np.array(adjacency_matrix['config']['value'])
        self.adjacency_matrix = adjacency_matrix
        self.graph_layer_sizes = graph_layer_sizes
        self.layer_sizes = layer_sizes
        self.A = spektral.utils.convolution.gcn_filter(adjacency_matrix)

        self.dense_tail = layer_sizes is not None
        self.graph_layers = [spektral.layers.GCNConv(size, activation='relu') for size in graph_layer_sizes[:-1]] 
        self.rescale_layer = tf.keras.layers.Rescaling(1, offset=-1e-6)

        self.flatten_layer = None
        self.dense_layers = None
        if self.dense_tail:
            self.graph_layers.append(spektral.layers.GCNConv(graph_layer_sizes[-1], activation='relu'))
            self.flatten_layer = tf.keras.layers.Flatten()
            self.dense_layers = [tf.keras.layers.Dense(size, activation='relu') for size in layer_sizes[:-1]]
            self.dense_layers.append(tf.keras.layers.Dense(layer_sizes[-1], activation='linear'))
        else:
            self.graph_layers.append(spektral.layers.GCNConv(graph_layer_sizes[-1], activation='sigmoid'))

    def call(self, x):
        for graph_layer in self.graph_layers:
            x = graph_layer([x, self.A])

        if self.dense_tail:
            x = self.flatten_layer(x)
            for layer in self.dense_layers:
                x = layer(x)
        else:
            x = self.rescale_layer(x)
        return x

    def debug_call(self, x):
        intermediate = [x]
        for graph_layer in self.graph_layers:
            x = graph_layer([x, self.A])
            intermediate.append(x)

        return intermediate

    def get_config(self):
        config = super(GraphElectronModel, self).get_config()
        config.update({
            'adjacency_matrix': self.adjacency_matrix,
            'graph_layer_sizes': self.graph_layer_sizes,
            'layer_sizes': self.layer_sizes
        })
        return config


    
class BaselinePhotonClassifier(tf.keras.Model):
    def __init__(self, input_size=None, layer_sizes=[1], name='baseline_photon_classifier_'):
        for size in layer_sizes:
            name += str(size) + '-'
        name = name[:-1]
        super().__init__(name=name)

        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layers = [tf.keras.layers.Dense(size, activation='sigmoid') for size in layer_sizes[:-1]]
        self.dense_layers.append(tf.keras.layers.Dense(layer_sizes[-1], activation='softmax'))

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        if input_size:
            self.compile(optimizer=self.optimizer, loss=self.loss)
            self.build((None, input_size))
    
    def call(self, x):
        x = self.flatten_layer(x)
        for layer in self.dense_layers:
            x = layer(x)
        return x
    

class ChannelPhotonModel(tf.keras.Model):
    def __init__(self, input_size=None, layer_sizes=[1], max=4, channel_size=16*16, name='channel_photon_model_'):
        for size in layer_sizes:
            name += str(size) + '-'
        name = name[:-1]
        super().__init__(name=name)

        self.model = BaselinePhotonModel(input_size=input_size, layer_sizes=layer_sizes, max=max)

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.MeanSquaredError()
        # self.loss = ScaledMeanSquaredError(delta=1 + 0.9 ** max)
        # self.loss = ScaledMeanSquaredError(delta=2)
        # self.loss = ChannelScaledMeanSquaredError(delta=2)
        if input_size:
            self.compile(optimizer=self.optimizer, loss=self.loss)
            self.build((None, channel_size, 1))
    
    def call(self, x):
        x = tf.keras.layers.TimeDistributed(self.model)(x)
        return x


class CustomMLPModel(tf.keras.Model):
    def __init__(self, input_size=None, layer_sizes=[1], classification=False, name='mlp_model_'):
        for size in layer_sizes:
            name += str(size) + '-'
        name = name[:-1]
        super().__init__(name=name)
        
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layers = [tf.keras.layers.Dense(size, activation='leaky_relu') for size in layer_sizes[:-1]]

        metrics = []
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        if classification: 
            self.dense_layers.append(tf.keras.layers.Dense(layer_sizes[-1], activation='sigmoid'))
            self.loss = tf.keras.losses.BinaryCrossentropy()
            metrics.append('AUC')
        else: 
            self.dense_layers.append(tf.keras.layers.Dense(layer_sizes[-1], activation='linear'))
            # self.loss = tf.keras.losses.MeanSquaredError()
            self.loss = tf.keras.losses.MeanSquaredError()

        if input_size:
            self.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics)
            self.build((None, input_size, 1))

    def call(self, x):
        x = self.flatten_layer(x)
        for layer in self.dense_layers:
            x = layer(x)
        
        return x

class RelativeError(tf.keras.losses.Loss):
    def __init__(self, e=1e-2):
        super().__init__()
        self.e = e
    
    def call(self, y, y_hat):
        y = tf.squeeze(y)
        epsilon = tf.ones_like(y) * self.e
        return tf.keras.losses.MeanAbsolutePercentageError()(y + epsilon, y_hat + epsilon)

class BinnedError(tf.keras.losses.Loss):
    def __init__(self, lam=0.9):
        super().__init__()
        self.lam = lam
    
    def call(self, y, y_hat):
        mse = tf.reduce_mean(tf.square(y - y_hat), axis=1)
        ve = 0 # tf.square(tf.reduce_sum(tf.abs(y), axis=1) - tf.reduce_sum(tf.abs(y_hat), axis=1))
        return tf.reduce_mean(self.lam * mse + (1 - self.lam) * ve)
    
class BinnedRelativeError(tf.keras.losses.Loss):
    def __init__(self, e=1e-1, lam=0.9999):
        super().__init__()
        self.e = e
        self.lam = lam
    
    def call(self, y, y_hat):
        re = tf.reduce_mean(tf.abs((y_hat + self.e) / (y + self.e)))
        ve = tf.reduce_mean(tf.square(tf.reduce_sum(tf.abs(y), axis=1) - tf.reduce_sum(tf.abs(y_hat), axis=1)))
        return self.lam * re + (1 - self.lam) * ve


class CustomMLPBinnedModel(tf.keras.Model):
    def __init__(self, input_size=None, layer_sizes=[1], name='mlp_binned_model_'):
        for size in layer_sizes:
            name += str(size) + '-'
        name = name[:-1]
        super().__init__(name=name)
        
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layers = [tf.keras.layers.Dense(size, activation='leaky_relu') for size in layer_sizes]

        metrics = []
        self.optimizer = tf.keras.optimizers.Adam()
        self.dense_layers.append(tf.keras.layers.Dense(input_size, activation='linear'))
        self.loss = BinnedError()

        if input_size:
            self.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics)
            self.build((None, input_size, 1))

    def call(self, x):
        x = self.flatten_layer(x)
        for layer in self.dense_layers:
            x = layer(x)
        
        return x



def tuner_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hp.Choice('units1', [150, 100, 75, 50, 25, 15, 10, 5]), activation='selu'))
    model.add(tf.keras.layers.Dense(hp.Choice('units2', [100, 75, 50, 25, 15, 10, 5]), activation='selu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(loss='mae')
    return model
    
def tuner_model_2(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hp.Choice('units1', [700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 75, 50, 25, 15, 10, 5]), activation='selu'))
    if hp.Boolean('layer2'):
        model.add(tf.keras.layers.Dense(hp.Choice('units2', [600, 550, 500, 450, 400, 350, 300, 250, 200, 100, 75, 50, 25, 15, 10, 5]), activation='selu'))
    if hp.Boolean('layer3'):
        model.add(tf.keras.layers.Dense(hp.Choice('units3', [500, 450, 400, 350, 300, 250, 200, 150, 100, 75, 50, 25, 15, 10, 5]), activation='selu'))
    if hp.Boolean('layer4'):
        model.add(tf.keras.layers.Dense(hp.Choice('units4', [400, 350, 300, 250, 200, 150, 100, 75, 50, 25, 15, 10, 5]), activation='selu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(loss='mae')
    return model


class MLPModel(tf.keras.Model):
    def __init__(self, input_size=None, output_size=1, classification=False, name='mlp_model'):
        super().__init__(name=name)

        dense1 = tf.keras.layers.Dense(512, activation='relu')
        dense2 = tf.keras.layers.Dense(128, activation='relu')
        dense3 = tf.keras.layers.Dense(32, activation='relu')
        dense4 = tf.keras.layers.Dense(output_size)

        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layers = [dense1, dense2, dense3, dense4]

        metrics = []
        self.optimizer = tf.keras.optimizers.Adam()
        if classification: 
            self.loss = tf.keras.losses.BinaryCrossentropy()
            metrics.append('AUC')
        else: 
            self.loss = tf.keras.losses.MeanAbsoluteError()
            # self.loss = tf.keras.losses.MeanSquaredError()
            # metrics.append(tf.keras.losses.MeanAbsoluteError())

        if input_size:
            self.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics)
            self.build((None, input_size, 1))

    def call(self, x):
        x = self.flatten_layer(x)
        for layer in self.dense_layers:
            x = layer(x)
        
        return x

    

class RNNModel(tf.keras.Model):
    def __init__(self, input_size, name='rnn_model'):
        super().__init__(name=name)

        dense1 = tf.keras.layers.Dense(64, activation='relu')
        dense2 = tf.keras.layers.Dense(16, activation='relu')
        dense3 = tf.keras.layers.Dense(1, activation='relu')

        self.rnn = tf.keras.layers.LSTM(64)
        self.dense_layers = [dense1, dense2, dense3]

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.MeanAbsoluteError()

    def call(self, x):
        x = self.rnn(x)
        for layer in self.dense_layers:
            x = layer(x)
        
        return x
    

class ConvModel2(tf.keras.Model):
    def __init__(self, input_size, name='conv_model_2'):
        super().__init__(name=name)

        dense1 = tf.keras.layers.Dense(256, activation='relu')
        dense2 = tf.keras.layers.Dense(64, activation='relu')
        dense3 = tf.keras.layers.Dense(16, activation='relu')
        dense4 = tf.keras.layers.Dense(1, activation='relu')

        self.conv_layer = tf.keras.layers.Convolution1D(filters=4, kernel_size=5, activation='relu')
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layers = [dense1, dense2, dense3, dense4]

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.MeanAbsoluteError()

        if input_size:
            self.compile(optimizer=self.optimizer, loss=self.loss)
            self.build((None, input_size, 1))

    def call(self, x):
        x = self.conv_layer(x)
        x = self.flatten_layer(x)
        for layer in self.dense_layers:
            x = layer(x)
        
        return x
    

class AttentionModel(tf.keras.Model):
    def __init__(self, input_size=None, output_size=1, classification=False, name='attention_model'):
        super().__init__(name=name)

        attention_size = 64
        self.K = tf.keras.layers.Dense(attention_size)
        self.Q = tf.keras.layers.Dense(attention_size)
        self.V = tf.keras.layers.Dense(attention_size)
        self.attention_layer = tf.keras.layers.Attention(use_scale=True)
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layers = [tf.keras.layers.Dense(64) for _ in range(2)]
        self.output_layer = tf.keras.layers.Dense(1, activation='linear')

        metrics = []
        self.optimizer = tf.keras.optimizers.Adam()
        if classification: 
            self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
            self.loss = tf.keras.losses.BinaryCrossentropy()
            metrics.append('AUC')
        else: 
            self.loss = tf.keras.losses.MeanAbsoluteError()

        if input_size:
            self.compile(optimizer=self.optimizer, loss=self.loss)
            self.build((None, input_size, 1))

    def call(self, x):
        Q = self.Q(x)
        V = self.V(x)
        QV_attention_seq = self.attention_layer([Q, V])
        Q_encoding = tf.keras.layers.GlobalAveragePooling1D()(Q)
        QV_attention = tf.keras.layers.GlobalAveragePooling1D()(QV_attention_seq)
        x = tf.keras.layers.Concatenate()([Q_encoding, QV_attention])

        for layer in self.dense_layers:
            x = layer(x)

        x = self.output_layer(x)

        return x  
    
# https://jeas.springeropen.com/articles/10.1186/s44147-023-00186-9
class ConvAttentionModel(tf.keras.Model):
    def __init__(self, input_size=None, output_size=1, classification=False, name='conv_attention_model'):
        super().__init__(name=name)

        self.block1 = [
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool1D(pool_size=2),
            tf.keras.layers.Attention()
        ]

        self.block2 = [
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool1D(pool_size=2),
            tf.keras.layers.Attention()
        ]

        self.block3 = [
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Attention()
        ]

        self.block4 = [
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.BatchNormalization()
        ]

        self.blocks = [self.block1, self.block2, self.block3, self.block4]

        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layers = [tf.keras.layers.Dense(128) for _ in range(2)]
        self.output_layer = tf.keras.layers.Dense(1, activation='linear')

        metrics = []
        self.optimizer = tf.keras.optimizers.Adam()
        if classification: 
            self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
            self.loss = tf.keras.losses.BinaryCrossentropy()
            metrics.append('AUC')
        else: 
            self.loss = tf.keras.losses.MeanAbsoluteError()

        if input_size:
            self.compile(optimizer=self.optimizer, loss=self.loss)
            self.build((None, input_size, 1))

    def call(self, x):
        for block in self.blocks:
            for layer in block:
                if isinstance(layer, tf.keras.layers.Attention):
                    x = layer([x, x])
                else:
                    x = layer(x)

        x = self.flatten_layer(x)
        
        for layer in self.dense_layers:
            x = layer(x)
        
        x = self.output_layer(x)

        return x
    

class ConvNoAttentionModel(tf.keras.Model):
    def __init__(self, input_size=None, output_size=1, classification=False, name='conv_no_attention_model'):
        super().__init__(name=name)

        self.block1 = [
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool1D(pool_size=2)
        ]

        self.block2 = [
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool1D(pool_size=2)
        ]

        self.block3 = [
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.BatchNormalization()
        ]

        self.block4 = [
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.BatchNormalization()
        ]

        self.blocks = [self.block1, self.block2, self.block3, self.block4]

        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layers = [tf.keras.layers.Dense(128) for _ in range(2)]
        self.output_layer = tf.keras.layers.Dense(output_size, activation='linear')

        metrics = []
        self.optimizer = tf.keras.optimizers.Adam()
        if classification: 
            self.output_layer = tf.keras.layers.Dense(output_size, activation='sigmoid')
            self.loss = tf.keras.losses.BinaryCrossentropy()
            metrics.append('AUC')
        else: 
            self.loss = tf.keras.losses.MeanAbsoluteError()

        if input_size:
            self.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics)
            self.build((None, input_size, 1))

    def call(self, x):
        for block in self.blocks:
            for layer in block:
                x = layer(x)

        x = self.flatten_layer(x)
        
        for layer in self.dense_layers:
            x = layer(x)
        
        x = self.output_layer(x)

        return x



class BaselineModel(tf.keras.Model):
    def __init__(self, input_size=700, output_size=1, classification=False, name='baseline_model'):
        super().__init__(name=name)
        self.model = self.build_model(input_shape=(input_size, 1))
        self.input_size = input_size
        self.output_size = output_size
        
        self.loss = tf.keras.losses.MeanAbsoluteError()
        self.optimizer = tf.keras.optimizers.Adam()

    def call(self, x):
        return self.model(x)

    def build_model(activation='relu', learning_rate = 1e-3, input_shape=(700, 1)):
        model = tf.keras.models.Sequential(name='mlp')
        model.add(tf.keras.layers.Input(shape=input_shape))
        model.add(tf.keras.layers.Flatten())
        initializer = tf.keras.initializers.HeNormal()
        model.add(tf.keras.layers.Dense(1000, kernel_initializer = initializer, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='mean_absolute_error', optimizer=optimizer)#, metrics=[tf.keras.metrics.RootMeanSquaredError()])
        model.build((1,) + input_shape)
        return model
    
class BaselineConvModel(tf.keras.Model):
    def __init__(self, input_size=None, output_size=1, name='baseline_model'):
        super().__init__(name=name)
        self.model = self.build_model()
        self.input_size = input_size
        self.output_size = output_size

    def call(self, x):
        return self.model(x)

    def build_model(self, n_hidden=1, n_neurons=50, activation='relu', learning_rate=1e-3, input_shape=(700, 1)):
        model = tf.keras.models.Sequential(name='two_layer_conv')
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

        model.add(tf.keras.layers.Conv1D(filters=8, kernel_size=3, padding='same', activation=activation))
        model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=activation))

        model.add(tf.keras.layers.Flatten())
        for layer in range(n_hidden):
            model.add(tf.keras.layers.Dense(n_neurons, activation=activation))

        model.add(tf.keras.layers.Dense(1, activation='linear'))
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        model.compile(loss='mean_absolute_error', optimizer=optimizer)

        return model
    
class HybridModel(tf.keras.Model):
    def __init__(self, input_size=None, output_size=1, classification=True, name='baseline_model'):
        super().__init__(name=name)
        self.MLP = MLPModel(output_size=32)
        self.CNN = build_vgg(length=input_size, width=64, name='vgg13', output_nums=32)
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

        metrics = []
        self.optimizer = tf.keras.optimizers.Adam()
        if classification: 
            self.loss = tf.keras.losses.BinaryCrossentropy()
            metrics.append('AUC')
        else: 
            self.loss = tf.keras.losses.MeanAbsoluteError()

        if input_size:
            self.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics)
            self.build((None, input_size, 1))
    
    def call(self, x):
        x1 = self.MLP(x)
        x2 = self.CNN(x)
        x = tf.concat([x1, x2], axis=1)
        x = self.output_layer(x)

        return x
    

class Autoencoder(tf.keras.Model):
    def __init__(self, input_size=None, encoder_layer_sizes=[1], decoder_layer_sizes=[700], name='mlp_encoder_'):
        for size in encoder_layer_sizes:
            name += str(size) + '-'
        for size in decoder_layer_sizes:
            name += str(size) + '-'
        name = name[:-1]
        super().__init__(name=name)
        
        self.flatten_layer = tf.keras.layers.Flatten()

        self.encoder = tf.keras.Sequential()
        for layer_size in encoder_layer_sizes[:-1]:
            self.encoder.add(tf.keras.layers.Dense(layer_size, activation='leaky_relu'))
        self.encoder.add(tf.keras.layers.Dense(encoder_layer_sizes[-1], activation='linear'))

        self.decoder = tf.keras.Sequential()
        for layer_size in decoder_layer_sizes[:-1]:
            self.decoder.add(tf.keras.layers.Dense(layer_size, activation='leaky_relu'))
        self.decoder.add(tf.keras.layers.Dense(decoder_layer_sizes[-1], activation='linear'))
        
        self.loss = RelativeError() # tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()

        if input_size:
            self.compile(optimizer=self.optimizer, loss=self.loss)
            self.build((None, input_size, 1))

    def call(self, x):
        x = self.flatten_layer(x)
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x
    
    def encode(self, x):
        x = self.flatten_layer(x)
        x = self.encoder(x)

        return np.array(np.expand_dims(x.numpy(), axis=-1), dtype=np.float16)


class VariationalAutoencoder(tf.keras.Model):
    def __init__(self, input_size=None, encoder_layer_sizes=[1], decoder_layer_sizes=[700], name='vae_'):
        for size in encoder_layer_sizes:
            name += str(size) + '-'
        for size in decoder_layer_sizes:
            name += str(size) + '-'
        name = name[:-1]
        super().__init__(name=name)
        
        self.flatten_layer = tf.keras.layers.Flatten()

        self.encoder = tf.keras.Sequential()
        for layer_size in encoder_layer_sizes[:-1]:
            self.encoder.add(tf.keras.layers.Dense(layer_size, activation='leaky_relu'))
        # Split the encoder output into two parameters: means and log variances
        self.encoder.add(tf.keras.layers.Dense(encoder_layer_sizes[-1], activation='linear'))

        self.decoder = tf.keras.Sequential()
        for layer_size in decoder_layer_sizes[:-1]:
            self.decoder.add(tf.keras.layers.Dense(layer_size, activation='leaky_relu'))
        self.decoder.add(tf.keras.layers.Dense(decoder_layer_sizes[-1], activation='linear'))
        
        self.optimizer = tf.keras.optimizers.Adam()

        if input_size:
            self.compile(optimizer=self.optimizer)
            self.build((None, input_size))
    
    def encode(self, x):
        x = self.flatten_layer(x)
        encoder_output = self.encoder(x)
        z_mean, z_log_var = tf.split(encoder_output, num_or_size_splits=2, axis=1)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
    
    def call(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_recon = self.decode(z)
        return x_recon, z_mean, z_log_var

    def compute_loss(self, x):
        x = self.flatten_layer(x)
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_recon = self.decode(z)
        recon_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, x_recon), axis=0)
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=0)
        print(recon_loss, kl_loss)
        total_loss = recon_loss + kl_loss
        return total_loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(data)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {'loss': loss}