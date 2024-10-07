from tensorflow.keras.layers import Input, Dense, Convolution2D, Convolution1D, Flatten, Reshape, MaxPooling1D, MaxPooling2D, LSTM, TimeDistributed, Attention, Rescaling
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, GlobalAveragePooling1D, GlobalAveragePooling2D, Concatenate, Activation, BatchNormalization, Dropout, MaxPool1D
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError, CategoricalCrossentropy, Loss, binary_crossentropy
from tensorflow.keras.initializers import RandomNormal, RandomUniform, HeNormal, HeUniform
from tensorflow import math as tfm
from tensorflow import expand_dims, unstack, stack, concat, gather
from tensorflow.nn import gelu, l2_normalize
import tensorflow_probability as tfp
tfd = tfp.distributions
from scipy.sparse import csr_matrix
import spektral
# import keras_tuner
import numpy as np
from vgg import *

from scipy.stats import wasserstein_distance
from matplotlib import pyplot as plt

from preprocessing import debug_print, dprint, MODEL_SAVE_PATH

'''
Save and load TF model weights
    model           : Tensorflow model
    name            : Save name

    return          : None
'''
def save_model_weights(model, name=None):
    if not name: name = model.name
    debug_print(['saving', name, 'weights to', MODEL_SAVE_PATH])
    model.save_weights(MODEL_SAVE_PATH + name + '.weights.h5')

def load_model_weights(model, name=None):
    if not name: name = model.name
    debug_print(['loading', name, 'weights from', MODEL_SAVE_PATH + name + '.weights.h5'])
    model.load_weights(MODEL_SAVE_PATH + name + '.weights.h5')
    return model

def save_model(model, name=None):
    if not name: name = model.name
    debug_print(['saving', name, 'to', MODEL_SAVE_PATH + name + '.keras'])
    model.save(MODEL_SAVE_PATH + name + '.keras')


def load(name):
    debug_print(['loading', name, 'from', MODEL_SAVE_PATH + name + '.keras'])
    return load_model(MODEL_SAVE_PATH + name + '.keras')


'''
Reset learned model weights to initialization
    model           : Tensorflow model

    return          : Reset model
'''
def reset_weights(model):
    debug_print(['resetting', model.name, 'weights'])
    for layer in model.layers:
        if isinstance(layer, (Convolution1D, Dense)):
            layer.set_weights([
                tf.keras.initializers.glorot_normal()(layer.weights[0].shape),
                tf.zeros(layer.weights[1].shape)  # Bias
            ])
    return model


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

        block1 = [Convolution1D(filters=32, kernel_size=3, strides=1) for _ in range(num_block1)]
        block2 = [Convolution1D(filters=32, kernel_size=3, strides=1) for _ in range(num_block2)]
        block3 = [Convolution1D(filters=64, kernel_size=5, strides=1) for _ in range(num_block3)]
        block4 = [Convolution1D(filters=64, kernel_size=5, strides=1) for _ in range(num_block4)]
        block5 = [Convolution1D(filters=64, kernel_size=7, strides=1) for _ in range(num_block5)]

        self.blocks = [block1, block2, block3, block4, block5]
        self.pooling_layer = MaxPooling1D(pool_size=2)
        self.flatten_layer = Flatten()
        self.dense_layers = [Dense(128, activation='relu') for _ in range(num_dense)]
        self.output_layer = Dense(1)

        self.optimizer = Adam()
        self.loss = MeanAbsoluteError()

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
        
        self.conv_layers = [Conv2D(num_filters, kernel_size, activation='leaky_relu', padding='valid') for _ in range(num_conv)]
        self.flatten_layer = Flatten()
        self.dense_layers = [Dense(size, activation='leaky_relu') for size in layer_sizes[:-1]]

        metrics = []
        self.optimizer = Adam(1e-5)
        if classification: 
            self.dense_layers.append(Dense(layer_sizes[-1], activation='sigmoid'))
            self.loss = BinaryCrossentropy()
            metrics.append('AUC')
        else: 
            self.dense_layers.append(Dense(layer_sizes[-1], activation='linear'))
            self.loss = MeanSquaredError()

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
        self.reshape_layer = Reshape((-1, input_size))
        self.td_heads = [tf.keras.Sequential([Dense(size, activation='leaky_relu') for size in head_sizes]) for _ in range(heads)]
        self.flatten_layer = Flatten()
        self.dense_layers = [
            Dense(512, activation='leaky_relu'),
            Dense(256, activation='leaky_relu')
        ]

        metrics = []
        self.optimizer = Adam()
        if classification: 
            self.dense_layers.append(Dense(layer_sizes[-1], activation='sigmoid'))
            self.loss = BinaryCrossentropy()
            metrics.append('AUC')
        else: 
            self.dense_layers.append(Dense(layer_sizes[-1], activation='linear'))
            # self.loss = MeanSquaredError()
            self.loss = MeanSquaredError()

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


class ChannelScaledMeanSquaredError(Loss):
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


class ScaledMeanSquaredError(Loss):
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

        self.flatten_layer = Flatten()
        self.dense_layers = [Dense(size, activation='sigmoid') for size in layer_sizes[:-1]]
        self.dense_layers.append(Dense(layer_sizes[-1], activation='sigmoid'))

        self.rescale_layer = Rescaling(max + 1, offset=-1e-2)
        self.optimizer = Adam()
        # self.loss = MeanSquaredError()
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


class MeanSquaredWassersteinLoss(Loss):
    def __init__(self):
        super().__init__()
    
    def call(self, y, y_hat):
        y_cumsum = tf.cumsum(y, axis=2)
        y_hat_cumsum = tf.cumsum(y_hat, axis=2)
        return tfm.reduce_mean((y_cumsum - y_hat_cumsum) ** 2)

class MeanSquaredEMDLoss3D(Loss):
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

        channel_mse = tfm.reduce_mean(tf.square(y_cumsum_xy - y_hat_cumsum_xy))
        sum_mse = tfm.reduce_mean(tf.square(y_cumsum_sum - y_hat_cumsum_sum))

        return channel_mse + sum_mse

    def get_config(self):
        return {
            'rows': self.rows,
            'cols': self.cols,
            'features': self.features
        }


class MeanSquaredEMDLoss(Loss):
    def __init__(self, rows=16, cols=16, features=700):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.features = features
    
    def call(self, y, y_hat):
        y_cumsum = tf.cumsum(y, axis=1)
        y_hat_cumsum = tf.cumsum(y_hat, axis=1)

        emd_loss = tfm.reduce_mean((y_cumsum - y_hat_cumsum) ** 2)
        square_error_loss = tfm.reduce_mean((y - y_hat) ** 2)

        return emd_loss # + square_error_loss

    def get_config(self):
        return {
            'rows': self.rows,
            'cols': self.cols,
            'features': self.features
        }


class MeanAbsoluteEMDLoss(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, y, y_hat):
        y_cumsum = tf.cumsum(y, axis=1)
        y_hat_cumsum = tf.cumsum(y_hat, axis=1)

        emd_loss = tfm.reduce_mean(tf.abs(y_cumsum - y_hat_cumsum))
        loss = tfm.reduce_mean(tf.abs(y - y_hat))

        return emd_loss + loss


class ElectronProbabilityLoss(Loss):
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
        self.dense_layers = [Dense(size, activation='relu', kernel_initializer=initializer) 
                             for size in layer_sizes[:-1]]
        self.dense_layers.append(Dense(layer_sizes[-1], activation='sigmoid', kernel_initializer=initializer))
        self.rescale_layer = Rescaling(1, offset=-1e-4)
    
    def call(self, x):
        for layer in self.dense_layers:
            x = TimeDistributed(layer)(x)
        x = self.rescale_layer(x)
        return x


class GraphElectronModel(tf.keras.Model):
    def __init__(self, adjacency_matrix, graph_layer_sizes=[1], layer_sizes=None, scale_factor=1, **kwargs):
        super(GraphElectronModel, self).__init__(**kwargs)
        
        if type(adjacency_matrix) == dict: # for deserialization
            adjacency_matrix = np.array(adjacency_matrix['config']['value'])
        self.adjacency_matrix = adjacency_matrix
        self.A = spektral.utils.gcn_filter(adjacency_matrix)
        self.graph_layer_sizes = graph_layer_sizes
        self.layer_sizes = layer_sizes

        self.dense_tail = layer_sizes is not None
        self.graph_layers = [spektral.layers.GCNConv(size, activation='relu') for size in graph_layer_sizes[:-1]] 
        self.scale_factor = scale_factor
        self.rescale_layer = Rescaling(scale_factor, offset=-1e-6)

        self.flatten_layer = None
        self.dense_layers = None
        if self.dense_tail:
            self.graph_layers.append(spektral.layers.GCNConv(graph_layer_sizes[-1], activation='relu'))
            self.flatten_layer = Flatten()
            self.dense_layers = [Dense(size, activation='relu') for size in layer_sizes[:-1]]
            self.dense_layers.append(Dense(layer_sizes[-1], activation='linear'))
        else:
            self.graph_layers.append(spektral.layers.GCNConv(graph_layer_sizes[-1], activation='sigmoid'))

    def call(self, x):
        # self.A_sparse = spektral.utils.sparse.sp_matrix_to_sp_tensor(self.adjacency_matrix)
        for graph_layer in self.graph_layers:
            # A_sparse = spektral.utils.sparse.sp_matrix_to_sp_tensor(csr_matrix(self.adjacency_matrix))
            x = graph_layer([x, self.A])

        if self.dense_tail:
            x = self.flatten_layer(x)
            for layer in self.dense_layers:
                x = layer(x)
        
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
            'layer_sizes': self.layer_sizes,
            'scale_factor': self.scale_factor
        })
        return config
    

class GraphElectronModel2(tf.keras.Model):
    def __init__(self, adjacency_matrix, graph_layer_sizes=[1], layer_sizes=None, scale_factor=1, **kwargs):
        super(GraphElectronModel2, self).__init__(**kwargs)
        
        if type(adjacency_matrix) == dict: # for deserialization
            adjacency_matrix = np.array(adjacency_matrix['config']['value'])
        self.adjacency_matrix = adjacency_matrix
        self.A = spektral.utils.gcn_filter(adjacency_matrix)
        self.graph_layer_sizes = graph_layer_sizes
        self.layer_sizes = layer_sizes

        self.dense_tail = layer_sizes is not None
        self.graph_layers = [MLPMessagePassing(size) for size in graph_layer_sizes[:-1]] 
        self.scale_factor = scale_factor
        self.rescale_layer = Rescaling(scale_factor, offset=-1e-6)

        self.flatten_layer = None
        self.dense_layers = None
        if self.dense_tail:
            self.graph_layers.append(MLPMessagePassing(graph_layer_sizes[-1]))
            self.flatten_layer = Flatten()
            self.dense_layers = [Dense(size, activation='relu') for size in layer_sizes[:-1]]
            self.dense_layers.append(Dense(layer_sizes[-1], activation='linear'))
        else:
            self.graph_layers.append(MLPMessagePassing(graph_layer_sizes[-1]))

    def call(self, inputs):
        x, a = inputs
        # self.A_sparse = spektral.utils.sparse.sp_matrix_to_sp_tensor(self.adjacency_matrix)
        for graph_layer in self.graph_layers:
            # A_sparse = spektral.utils.sparse.sp_matrix_to_sp_tensor(csr_matrix(self.adjacency_matrix))
            x = graph_layer([x, a])

        if self.dense_tail:
            x = self.flatten_layer(x)
            for layer in self.dense_layers:
                x = layer(x)
        
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
            'layer_sizes': self.layer_sizes,
            'scale_factor': self.scale_factor
        })
        return config
    


class MLPMessagePassing(spektral.layers.MessagePassing):
    def __init__(self, channels, mlp_hidden_units=64, **kwargs):
        # Initialize the parent class and the MLP
        super().__init__(**kwargs)
        self.channels = channels
        self.mlp = tf.keras.Sequential([
            Dense(mlp_hidden_units, activation='relu'),
            Dense(channels, activation='relu')  # Output dimension of message should match channels
        ])
    
    def build(self, input_shape):
        super().build(input_shape)
    
    def message(self, x_i, x_j, e_ij=None):
        if e_ij is not None:
            message_input = tf.concat([x_i, x_j, e_ij], axis=-1)
        else:
            message_input = tf.concat([x_i, x_j], axis=-1)
        return self.mlp(message_input)
    
    def aggregate(self, messages, index):
        return tf.math.unsorted_segment_mean(messages, index, num_segments=tf.reduce_max(index) + 1)
    
    def update(self, embeddings):
        return embeddings
    
    def call(self, inputs):
        x, a = inputs[:2]
        e = inputs[2] if len(inputs) == 3 else None

        # if not isinstance(a, tf.sparse.SparseTensor):
        #     a = tf.sparse.from_dense(a)
        
        # Perform message passing using the parent class's propagate method
        return self.propagate(x=x, a=a)
    

class ConvGraphElectronModel(tf.keras.Model):
    def __init__(self, graph_layer_sizes, layer_sizes, **kwargs):
        super(ConvGraphElectronModel, self).__init__(**kwargs)
        
        self.graph_layer_sizes = graph_layer_sizes
        self.layer_sizes = layer_sizes

        self.graph_layers = [Conv2D(kernel_size=(3, 3), filters=size, activation='selu', padding='valid') for size in graph_layer_sizes]
        self.flatten_layer = Flatten()
        self.dense_layers = [Dense(size, activation='selu') for size in layer_sizes[:-1]]
        self.dense_layers.append(Dense(layer_sizes[-1], activation='linear'))
    
    def call(self, x):
        for graph_layer in self.graph_layers:
            x = graph_layer(x)
        
        x = self.flatten_layer(x)
        for layer in self.dense_layers:
            x = layer(x)
        
        return x
    
    def get_config(self):
        config = super(ConvGraphElectronModel, self).get_config()
        config.update({
            'graph_layer_sizes': self.graph_layer_sizes,
            'layer_sizes': self.layer_sizes,
        })
        return config
    

class Conv3DGraphElectronModel(tf.keras.Model):
    def __init__(self, graph_layer_sizes, layer_sizes, **kwargs):
        super(Conv3DGraphElectronModel, self).__init__(**kwargs)
        
        self.graph_layer_sizes = graph_layer_sizes
        self.layer_sizes = layer_sizes

        self.graph_layers = [Conv3D(kernel_size=(3, 3, 4), filters=size, activation='selu', padding='valid') for size in graph_layer_sizes]
        self.flatten_layer = Flatten()
        self.dense_layers = [Dense(size, activation='selu') for size in layer_sizes[:-1]]
        self.dense_layers.append(Dense(layer_sizes[-1], activation='linear'))
    
    def call(self, x):
        for graph_layer in self.graph_layers:
            x = graph_layer(x)
        
        x = self.flatten_layer(x)
        for layer in self.dense_layers:
            x = layer(x)
        
        return x
    
    def get_config(self):
        config = super(Conv3DGraphElectronModel, self).get_config()
        config.update({
            'graph_layer_sizes': self.graph_layer_sizes,
            'layer_sizes': self.layer_sizes,
        })
        return config

    

class MLPElectronModel(tf.keras.Model):
    def __init__(self, layer_sizes=[700], name='mlp_electron_model_', *args, **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.dense_layers = [Dense(size, activation='sigmoid') 
                             for size in layer_sizes[:-1]]
        self.dense_layers.append(Dense(layer_sizes[-1], activation='linear'))
    
    def call(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        return x


    
class BaselinePhotonClassifier(tf.keras.Model):
    def __init__(self, input_size=None, layer_sizes=[1], name='baseline_photon_classifier_'):
        for size in layer_sizes:
            name += str(size) + '-'
        name = name[:-1]
        super().__init__(name=name)

        self.flatten_layer = Flatten()
        self.dense_layers = [Dense(size, activation='sigmoid') for size in layer_sizes[:-1]]
        self.dense_layers.append(Dense(layer_sizes[-1], activation='softmax'))

        self.optimizer = Adam()
        self.loss = CategoricalCrossentropy()
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

        self.optimizer = Adam()
        self.loss = MeanSquaredError()
        # self.loss = ScaledMeanSquaredError(delta=1 + 0.9 ** max)
        # self.loss = ScaledMeanSquaredError(delta=2)
        # self.loss = ChannelScaledMeanSquaredError(delta=2)
        if input_size:
            self.compile(optimizer=self.optimizer, loss=self.loss)
            self.build((None, channel_size, 1))
    
    def call(self, x):
        x = TimeDistributed(self.model)(x)
        return x


class CustomMLPModel(tf.keras.Model):
    def __init__(self, input_size=None, layer_sizes=[1], classification=False, name='mlp_model_'):
        for size in layer_sizes:
            name += str(size) + '-'
        name = name[:-1]
        super().__init__(name=name)
        
        self.flatten_layer = Flatten()
        self.dense_layers = [Dense(size, activation='leaky_relu') for size in layer_sizes[:-1]]

        metrics = []
        self.optimizer = Adam(learning_rate=1e-4)
        if classification: 
            self.dense_layers.append(Dense(layer_sizes[-1], activation='sigmoid'))
            self.loss = BinaryCrossentropy()
            metrics.append('AUC')
        else: 
            self.dense_layers.append(Dense(layer_sizes[-1], activation='linear'))
            # self.loss = MeanSquaredError()
            self.loss = MeanSquaredError()

        if input_size:
            self.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics)
            self.build((None, input_size, 1))

    def call(self, x):
        x = self.flatten_layer(x)
        for layer in self.dense_layers:
            x = layer(x)
        
        return x

class RelativeError(Loss):
    def __init__(self, e=1e-2):
        super().__init__()
        self.e = e
    
    def call(self, y, y_hat):
        y = tf.squeeze(y)
        epsilon = tf.ones_like(y) * self.e
        return MeanAbsolutePercentageError()(y + epsilon, y_hat + epsilon)

class BinnedError(Loss):
    def __init__(self, lam=0.9):
        super().__init__()
        self.lam = lam
    
    def call(self, y, y_hat):
        mse = tf.reduce_mean(tf.square(y - y_hat), axis=1)
        ve = 0 # tf.square(tf.reduce_sum(tf.abs(y), axis=1) - tf.reduce_sum(tf.abs(y_hat), axis=1))
        return tf.reduce_mean(self.lam * mse + (1 - self.lam) * ve)
    
class BinnedRelativeError(Loss):
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
        
        self.flatten_layer = Flatten()
        self.dense_layers = [Dense(size, activation='leaky_relu') for size in layer_sizes]

        metrics = []
        self.optimizer = Adam()
        self.dense_layers.append(Dense(input_size, activation='linear'))
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
    model.add(Dense(hp.Choice('units1', [150, 100, 75, 50, 25, 15, 10, 5]), activation='selu'))
    model.add(Dense(hp.Choice('units2', [100, 75, 50, 25, 15, 10, 5]), activation='selu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae')
    return model
    
def tuner_model_2(hp):
    model = tf.keras.Sequential()
    model.add(Dense(hp.Choice('units1', [700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 75, 50, 25, 15, 10, 5]), activation='selu'))
    if hp.Boolean('layer2'):
        model.add(Dense(hp.Choice('units2', [600, 550, 500, 450, 400, 350, 300, 250, 200, 100, 75, 50, 25, 15, 10, 5]), activation='selu'))
    if hp.Boolean('layer3'):
        model.add(Dense(hp.Choice('units3', [500, 450, 400, 350, 300, 250, 200, 150, 100, 75, 50, 25, 15, 10, 5]), activation='selu'))
    if hp.Boolean('layer4'):
        model.add(Dense(hp.Choice('units4', [400, 350, 300, 250, 200, 150, 100, 75, 50, 25, 15, 10, 5]), activation='selu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae')
    return model


class SquareRootError(Loss):
    def __init__(self):
        super().__init__()
    
    def call(self, y, y_hat):
        return tf.reduce_mean(tf.sqrt(tf.abs(y - y_hat) + 1))
    
class PowError(tf.keras.losses.Loss):
    def __init__(self, p=1.0, name="mean_absolute_error"):
        super(PowError, self).__init__(name=name)
        self.p = p

    def call(self, y_true, y_pred):
        abs_diff = tf.abs(y_true - y_pred)
        abs_diff_pow = tf.pow(abs_diff, self.p)
        return tf.reduce_mean(abs_diff)
    
def mean_absolute_error(y_true, y_pred):
    abs_diff = tf.abs(y_true - y_pred)
    abs_diff_pow = tf.pow(abs_diff, 1.0)
    return tf.reduce_mean(abs_diff_pow)

def mean_square_root_error(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.abs(y_true - y_pred)))

POW = 0.8
def mean_absolute_power_error(y_true, y_pred):
    abs_diff = tf.abs(y_true - y_pred)
    abs_diff_pow = tf.pow(abs_diff, POW)
    return tf.reduce_mean(abs_diff_pow)

def mean_log_error(y_true, y_pred):
    return tf.reduce_mean(tf.math.log(tf.abs(y_true - y_pred) + 1))
    

class LogError(Loss):
    def __init__(self):
        super().__init__()
    
    def call(self, y, y_hat):
        return tf.reduce_mean(tf.math.log(tf.abs(y - y_hat) + 1))
    

class CustomMeanAbsoluteError(tf.keras.losses.Loss):
    def __init__(self, name="mean_absolute_error"):
        super(CustomMeanAbsoluteError, self).__init__(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name=name)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        abs_diff = tf.abs(y_true - y_pred)
        return tf.reduce_mean(abs_diff)


class MLPModel(tf.keras.Model):
    def __init__(self, name='mlp_model'):
        super().__init__(name=name)

        initializer = HeNormal()
        dense1 = Dense(700, kernel_initializer=initializer, activation='selu')
        dense2 = Dense(512, kernel_initializer=initializer, activation='selu')
        dense3 = Dense(128, kernel_initializer=initializer, activation='selu')
        dense4 = Dense(1, kernel_initializer=initializer, activation='linear')

        self.dense_layers = [dense1, dense2, dense3, dense4]

    def call(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        
        return x
    

def confidence_mae_loss(y_true, output):
    confidence_exponent = 3
    confidence_threshold = 100

    y_pred, confidence = output[:, 0], output[:, 1]

    x = tf.abs(y_true - y_pred)
    return (1 - confidence) * x + confidence * tf.pow(x / confidence_threshold, confidence_exponent) * confidence_threshold


def eval_mae_loss(y_true, output):
    y_pred = output[:, 0]
    x = tf.abs(y_true - y_pred)
    return x


class MLPConfidenceModel(tf.keras.Model):
    def __init__(self, name='mlp_confidence_model'):
        super().__init__(name=name)

        initializer = HeNormal()
        dense1 = Dense(700, kernel_initializer=initializer, activation='selu')
        dense2 = Dense(512, kernel_initializer=initializer, activation='selu')
        dense3 = Dense(128, kernel_initializer=initializer, activation='selu')
        
        self.linear_ouput = Dense(1, kernel_initializer=initializer, activation='linear')
        self.confidence_output = Dense(1, kernel_initializer=initializer, activation='sigmoid')

        self.dense_layers = [dense1, dense2, dense3]

    def call(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        
        linear_output = self.linear_ouput(x)
        confidence_output = self.confidence_output(x)

        concat_output = tf.concat([linear_output, confidence_output], axis=-1)

        return concat_output


def skewnormal_pdf(x, params):
    mu, sigma, alpha = params[:, 0], params[:, 1], params[:, 2]

    standard_normal = tfd.Normal(loc=0.0, scale=1.0)
    
    z = (x - mu) / sigma
    
    pdf_normal = standard_normal.prob(z) # standard normal PDF (φ)
    cdf_normal = standard_normal.cdf(alpha * z) # standard normal CDF (Φ)

    skew_pdf = (2 / sigma) * pdf_normal * cdf_normal
    
    return skew_pdf

@tf.keras.utils.register_keras_serializable()
def skewnormal_pdf_loss(y_true, output):
    pdf_skew_normal = skewnormal_pdf(y_true / 1000, output)
    log_pdf_skew_normal = tf.math.log(pdf_skew_normal + 1e-12)

    return -tf.reduce_mean(log_pdf_skew_normal)

def normal_pdf(x, params):
    mu, sigma = params[:, 0], params[:, 1]
    normal_dist = tfd.Normal(loc=mu, scale=sigma)
    pdf_normal = normal_dist.prob(x)

    return pdf_normal

@tf.keras.utils.register_keras_serializable()
def normal_pdf_loss(y_true, output):
    pdf_normal = normal_pdf(y_true / 1000, output)
    log_pdf_normal = tf.math.log(pdf_normal + 1e-12)

    return -tf.reduce_mean(log_pdf_normal)

@tf.keras.utils.register_keras_serializable()
def mu_loss(y_true, output):
    mu = output[:, 0] * 1000
    return tf.reduce_mean(tf.abs(y_true - mu))


class MLPDistributionModel(tf.keras.Model):
    def __init__(self, name='mlp_distribution_model', **kwargs):
        super().__init__(name=name, **kwargs)

        initializer = RandomNormal()
        small_initializer = RandomNormal(1e-6)
        dense1 = Dense(700, kernel_initializer=initializer, activation='selu')
        dense2 = Dense(512, kernel_initializer=initializer, activation='selu')
        dense3 = Dense(128, kernel_initializer=initializer, activation='selu')
        
        self.mean_output = Dense(1, kernel_initializer=initializer, activation='linear')
        self.std_output = Dense(1, kernel_initializer=small_initializer, bias_initializer=tf.keras.initializers.Constant(1), activation='softplus')
        self.skew_output = Dense(1, kernel_initializer=initializer, activation='tanh')

        self.dense_layers = [dense1, dense2, dense3]
    
    def call(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        
        mean_output = self.mean_output(x)
        std_output = self.std_output(x) + 1e-5
        skew_output = tf.zeros_like(std_output) # self.skew_output(x)

        concat_output = tf.concat([mean_output, std_output, skew_output], axis=-1)

        return concat_output
    

    

class RNNModel(tf.keras.Model):
    def __init__(self, input_size, name='rnn_model'):
        super().__init__(name=name)

        dense1 = Dense(64, activation='relu')
        dense2 = Dense(16, activation='relu')
        dense3 = Dense(1, activation='relu')

        self.rnn = LSTM(64)
        self.dense_layers = [dense1, dense2, dense3]

        self.optimizer = Adam()
        self.loss = MeanAbsoluteError()

    def call(self, x):
        x = self.rnn(x)
        for layer in self.dense_layers:
            x = layer(x)
        
        return x
    

class ConvModel2(tf.keras.Model):
    def __init__(self, input_size, name='conv_model_2'):
        super().__init__(name=name)

        dense1 = Dense(256, activation='relu')
        dense2 = Dense(64, activation='relu')
        dense3 = Dense(16, activation='relu')
        dense4 = Dense(1, activation='relu')

        self.conv_layer = Convolution1D(filters=4, kernel_size=5, activation='relu')
        self.flatten_layer = Flatten()
        self.dense_layers = [dense1, dense2, dense3, dense4]

        self.optimizer = Adam()
        self.loss = MeanAbsoluteError()

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
        self.K = Dense(attention_size)
        self.Q = Dense(attention_size)
        self.V = Dense(attention_size)
        self.attention_layer = Attention(use_scale=True)
        self.flatten_layer = Flatten()
        self.dense_layers = [Dense(64) for _ in range(2)]
        self.output_layer = Dense(1, activation='linear')

        metrics = []
        self.optimizer = Adam()
        if classification: 
            self.output_layer = Dense(1, activation='sigmoid')
            self.loss = BinaryCrossentropy()
            metrics.append('AUC')
        else: 
            self.loss = MeanAbsoluteError()

        if input_size:
            self.compile(optimizer=self.optimizer, loss=self.loss)
            self.build((None, input_size, 1))

    def call(self, x):
        Q = self.Q(x)
        V = self.V(x)
        QV_attention_seq = self.attention_layer([Q, V])
        Q_encoding = GlobalAveragePooling1D()(Q)
        QV_attention = GlobalAveragePooling1D()(QV_attention_seq)
        x = Concatenate()([Q_encoding, QV_attention])

        for layer in self.dense_layers:
            x = layer(x)

        x = self.output_layer(x)

        return x  
    
# https://jeas.springeropen.com/articles/10.1186/s44147-023-00186-9
class ConvAttentionModel(tf.keras.Model):
    def __init__(self, input_size=None, output_size=1, classification=False, name='conv_attention_model'):
        super().__init__(name=name)

        self.block1 = [
            Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal'),
            Activation('relu'),
            BatchNormalization(),
            MaxPool1D(pool_size=2),
            Attention()
        ]

        self.block2 = [
            Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal'),
            Activation('relu'),
            BatchNormalization(),
            MaxPool1D(pool_size=2),
            Attention()
        ]

        self.block3 = [
            Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal'),
            Activation('relu'),
            BatchNormalization(),
            Attention()
        ]

        self.block4 = [
            Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal'),
            Activation('relu'),
            BatchNormalization()
        ]

        self.blocks = [self.block1, self.block2, self.block3, self.block4]

        self.flatten_layer = Flatten()
        self.dense_layers = [Dense(128) for _ in range(2)]
        self.output_layer = Dense(1, activation='linear')

        metrics = []
        self.optimizer = Adam()
        if classification: 
            self.output_layer = Dense(1, activation='sigmoid')
            self.loss = BinaryCrossentropy()
            metrics.append('AUC')
        else: 
            self.loss = MeanAbsoluteError()

        if input_size:
            self.compile(optimizer=self.optimizer, loss=self.loss)
            self.build((None, input_size, 1))

    def call(self, x):
        for block in self.blocks:
            for layer in block:
                if isinstance(layer, Attention):
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
            Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal'),
            Activation('relu'),
            BatchNormalization(),
            MaxPool1D(pool_size=2)
        ]

        self.block2 = [
            Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal'),
            Activation('relu'),
            BatchNormalization(),
            MaxPool1D(pool_size=2)
        ]

        self.block3 = [
            Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal'),
            Activation('relu'),
            BatchNormalization()
        ]

        self.block4 = [
            Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal'),
            Activation('relu'),
            BatchNormalization()
        ]

        self.blocks = [self.block1, self.block2, self.block3, self.block4]

        self.flatten_layer = Flatten()
        self.dense_layers = [Dense(128) for _ in range(2)]
        self.output_layer = Dense(output_size, activation='linear')

        metrics = []
        self.optimizer = Adam()
        if classification: 
            self.output_layer = Dense(output_size, activation='sigmoid')
            self.loss = BinaryCrossentropy()
            metrics.append('AUC')
        else: 
            self.loss = MeanAbsoluteError()

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
        
        self.loss = MeanAbsoluteError()
        self.optimizer = Adam()

    def call(self, x):
        return self.model(x)

    def build_model(activation='relu', learning_rate = 1e-3, input_shape=(700, 1)):
        model = Sequential(name='mlp')
        model.add(Input(shape=input_shape))
        model.add(Flatten())
        initializer = tf.keras.initializers.HeNormal()
        model.add(Dense(1000, kernel_initializer = initializer, activation='relu'))
        model.add(Dense(1, activation='linear'))
        optimizer = Adam(learning_rate=learning_rate)
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
        model = Sequential(name='two_layer_conv')
        model.add(Input(input_shape=input_shape))

        model.add(Conv1D(filters=8, kernel_size=3, padding='same', activation=activation))
        model.add(Conv1D(filters=16, kernel_size=5, padding='same', activation=activation))

        model.add(Flatten())
        for layer in range(n_hidden):
            model.add(Dense(n_neurons, activation=activation))

        model.add(Dense(1, activation='linear'))
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='mean_absolute_error', optimizer=optimizer)

        return model
    
class HybridModel(tf.keras.Model):
    def __init__(self, input_size=None, output_size=1, classification=True, name='baseline_model'):
        super().__init__(name=name)
        self.MLP = MLPModel(output_size=32)
        self.CNN = build_vgg(length=input_size, width=64, name='vgg13', output_nums=32)
        self.output_layer = Dense(1, activation='sigmoid')

        metrics = []
        self.optimizer = Adam()
        if classification: 
            self.loss = BinaryCrossentropy()
            metrics.append('AUC')
        else: 
            self.loss = MeanAbsoluteError()

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
        
        self.flatten_layer = Flatten()

        self.encoder = tf.keras.Sequential()
        for layer_size in encoder_layer_sizes[:-1]:
            self.encoder.add(Dense(layer_size, activation='leaky_relu'))
        self.encoder.add(Dense(encoder_layer_sizes[-1], activation='linear'))

        self.decoder = tf.keras.Sequential()
        for layer_size in decoder_layer_sizes[:-1]:
            self.decoder.add(Dense(layer_size, activation='leaky_relu'))
        self.decoder.add(Dense(decoder_layer_sizes[-1], activation='linear'))
        
        self.loss = RelativeError() # MeanSquaredError()
        self.optimizer = Adam()

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
        
        self.flatten_layer = Flatten()

        self.encoder = tf.keras.Sequential()
        for layer_size in encoder_layer_sizes[:-1]:
            self.encoder.add(Dense(layer_size, activation='leaky_relu'))
        # Split the encoder output into two parameters: means and log variances
        self.encoder.add(Dense(encoder_layer_sizes[-1], activation='linear'))

        self.decoder = tf.keras.Sequential()
        for layer_size in decoder_layer_sizes[:-1]:
            self.decoder.add(Dense(layer_size, activation='leaky_relu'))
        self.decoder.add(Dense(decoder_layer_sizes[-1], activation='linear'))
        
        self.optimizer = Adam()

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
        recon_loss = tf.reduce_mean(binary_crossentropy(x, x_recon), axis=0)
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