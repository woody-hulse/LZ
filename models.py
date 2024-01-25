import tensorflow as tf
from vgg import *


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
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
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
    

class MLPModel(tf.keras.Model):
    def __init__(self, input_size=None, output_size=1, name='mlp_model'):
        super().__init__(name=name)

        dense1 = tf.keras.layers.Dense(256, activation='relu')
        dense2 = tf.keras.layers.Dense(64, activation='relu')
        dense3 = tf.keras.layers.Dense(16, activation='relu')
        dense4 = tf.keras.layers.Dense(output_size, activation='sigmoid')

        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layers = [dense1, dense2, dense3, dense4]

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

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
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

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
        dense4 = tf.keras.layers.Dense(1, activation='sigmoid')

        self.conv_layer = tf.keras.layers.Convolution1D(filters=4, kernel_size=5, activation='relu')
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layers = [dense1, dense2, dense3, dense4]

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def call(self, x):
        x = self.conv_layer(x)
        x = self.flatten_layer(x)
        for layer in self.dense_layers:
            x = layer(x)
        
        return x
    

class BaselineModel(tf.keras.Model):
    def __init__(self, input_size=None, output_size=1, name='baseline_model'):
        super().__init__(name=name)
        self.model = self.build_model()
        self.input_size = input_size
        self.output_size = output_size

    def call(self, x):
        return self.model(x)

    def build_model(self, n_hidden=1, n_neurons=50, activation='relu', learning_rate=1e-3, input_shape=(700, 1)):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

        model.add(tf.keras.layers.Conv1D(filters=8, kernel_size=3, padding='same', activation=activation))
        model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=activation))

        model.add(tf.keras.layers.Flatten())
        for layer in range(n_hidden):
            model.add(tf.keras.layers.Dense(n_neurons, activation=activation))

        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["AUC"])

        return model
