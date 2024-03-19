import tensorflow as tf
import numpy as np

class VectorQuantizer(tf.keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


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
        
        self.loss = tf.keras.losses.MeanSquaredError()
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

        return np.expand_dims(x.numpy(), axis=-1)


def get_vqvae(input_size=700, num_embeddings=64, encoder_layer_sizes=[1], decoder_layer_sizes=[700]):
    latent_dim = encoder_layer_sizes[-1]
    autoencoder = Autoencoder(input_size=700, encoder_layer_sizes=encoder_layer_sizes, decoder_layer_sizes=decoder_layer_sizes)
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    encoder = autoencoder.encoder
    decoder = autoencoder.decoder
    inputs = tf.keras.Input(shape=(input_size, 1))
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)
    return tf.keras.Model(inputs, reconstructions, name="vq_vae")

class VQVariationalAutoencoder(tf.keras.Model):
    def __init__(self, input_size=700, num_embeddings=64, encoder_layer_sizes=[1], decoder_layer_sizes=[700], name='vqvae'):
        super().__init__(name=name)
        latent_dim = encoder_layer_sizes[-1]
        autoencoder = Autoencoder(input_size=700, encoder_layer_sizes=encoder_layer_sizes, decoder_layer_sizes=decoder_layer_sizes)
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder
        self.flatten_layer = tf.keras.layers.Flatten()

        self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
        self.build((None, input_size, 1))
    
    def call(self, x):
        x = self.encoder(x)
        x = self.vq_layer(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.flatten_layer(x)
        x = self.encoder(x)

        return np.expand_dims(x.numpy(), axis=-1)
    
    def vq_encode(self, x):
        x = self.flatten_layer(x)
        x = self.encoder(x)
        x = self.vq_layer(x)

        return np.expand_dims(x.numpy(), axis=-1)