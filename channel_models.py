import tensorflow as tf
import numpy as np

class BaselineMLP(tf.keras.Model):
    def __init__(self, input_size=None, output_size=1, classification=False, name='baseline_mlp'):
        super().__init__(name=name)

        self.dense_layers = [tf.keras.layers.Dense((5 - i) ** 3, activation='relu') for i in range(4)]
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
            self.compile(optimizer=self.optimizer, loss=self.loss)
            self.build((None, input_size, 1))
    
    def call(self, x):
        x = tf.keras.layers.Flatten()(x)
        for layer in self.dense_layers:
            x = layer(x)
        
        x = self.output_layer(x)

        return x
    

class BaselineConv(tf.keras.Model):
    def __init__(self, input_size=None, output_size=1, classification=False, name='baseline_conv'):
        super().__init__(name=name)

        self.conv_layers = [
            tf.keras.layers.Convolution2D(filters=32, kernel_size=5, padding='valid', activation='relu'),
            tf.keras.layers.Convolution2D(filters=32, kernel_size=5, padding='valid', activation='relu'),
            tf.keras.layers.Convolution2D(filters=64, kernel_size=3, padding='valid', activation='relu'),
            tf.keras.layers.Convolution2D(filters=64, kernel_size=5, padding='valid', activation='relu')
        ]
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layers = [tf.keras.layers.Dense((3 - i) ** 3, activation='relu') for i in range(4)]
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
            self.compile(optimizer=self.optimizer, loss=self.loss)
            self.build((None, input_size, 1))
    
    def call(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten_layer(x)
        for layer in self.dense_layers:
            x = layer(x)
        
        x = self.output_layer(x)

        return x
    

class PODMLP(tf.keras.Model):
    def __init__(self, input_size=None, output_size=1, classification=False, name='pod_mlp'):
        super().__init__(name=name)

        self.dense_layers = [tf.keras.layers.Dense((5 - i) ** 3, activation='relu') for i in range(4)]
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
            self.compile(optimizer=self.optimizer, loss=self.loss)
            self.build((None, input_size, 1))
    
    @staticmethod
    def POD(self, data, num_modes=50):
        spatial_data = []
        temporal_data = []
        for channels in data:
            u, s, v = np.linalg.svd(channels, full_matrices=True)
            epsilon = s[num_modes + 1] / s[num_modes]
            energy = np.sum(s[:num_modes]) / np.sum(s)
            s = np.diag(s[:num_modes])
            v = v[:, :num_modes]

            spatial_modes = u[:, :num_modes]
            temporal_modes = s @ v.T

            spatial_data.append(spatial_modes)
            temporal_data.append(temporal_modes)
        
        return np.array(spatial_data), np.array(temporal_data)

    def call(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        
        x = self.output_layer(x)

        return x


class BaselineGNN(tf.keras.Model):
    def __init__(self, num_channels, output_size=1, classification=False, name='baseline_gnn'):
        super().__init__(name=name)
        self.num_channels = num_channels

        self.graph_conv1 = GraphConvolution(64)
        self.graph_conv2 = GraphConvolution(32)

        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def call(self, inputs, adjacency_matrix):
        # Graph convolutional layers
        x = self.graph_conv1([inputs, adjacency_matrix])
        x = self.graph_conv2([x, adjacency_matrix])

        # Flatten the output
        x = self.flatten(x)

        # Dense layers
        x = self.dense1(x)
        output = self.dense2(x)

        return output


def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(tf.keras.layers.BatchNormalization())
        fnn_layers.append(tf.keras.layers.Dropout(dropout_rate))
        fnn_layers.append(tf.keras.layers.Dense(units, activation=tf.nn.gelu))

    return tf.keras.keras.Sequential(fnn_layers, name=name)


class GraphConvolution(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_units,
        dropout_rate=0.2,
        aggregation_type="mean",
        combination_type="concat",
        normalize=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.normalize = normalize

        self.ffn_prepare = create_ffn(hidden_units, dropout_rate)
        if self.combination_type == "gated":
            self.update_fn = tf.keras.layers.GRU(
                units=hidden_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                dropout=dropout_rate,
                return_state=True,
                recurrent_dropout=dropout_rate,
            )
        else:
            self.update_fn = create_ffn(hidden_units, dropout_rate)

    def prepare(self, node_repesentations, weights=None):
        # node_repesentations shape is [num_edges, embedding_dim].
        messages = self.ffn_prepare(node_repesentations)
        if weights is not None:
            messages = messages * tf.expand_dims(weights, -1)
        return messages

    def aggregate(self, node_indices, neighbour_messages, node_repesentations):
        # node_indices shape is [num_edges].
        # neighbour_messages shape: [num_edges, representation_dim].
        # node_repesentations shape is [num_nodes, representation_dim]
        num_nodes = node_repesentations.shape[0]
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "max":
            aggregated_message = tf.math.unsorted_segment_max(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")

        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        # node_repesentations shape is [num_nodes, representation_dim].
        # aggregated_messages shape is [num_nodes, representation_dim].
        if self.combination_type == "gru":
            # Create a sequence of two elements for the GRU layer.
            h = tf.stack([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "concat":
            # Concatenate the node_repesentations and aggregated_messages.
            h = tf.concat([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "add":
            # Add node_repesentations and aggregated_messages.
            h = node_repesentations + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        # Apply the processing function.
        node_embeddings = self.update_fn(h)
        if self.combination_type == "gru":
            node_embeddings = tf.unstack(node_embeddings, axis=1)[-1]

        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings

    def call(self, inputs):
        """Process the inputs to produce the node_embeddings.

        inputs: a tuple of three elements: node_repesentations, edges, edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """

        node_repesentations, edges, edge_weights = inputs
        # Get node_indices (source) and neighbour_indices (target) from edges.
        node_indices, neighbour_indices = edges[0], edges[1]
        # neighbour_repesentations shape is [num_edges, representation_dim].
        neighbour_repesentations = tf.gather(node_repesentations, neighbour_indices)

        # Prepare the messages of the neighbours.
        neighbour_messages = self.prepare(neighbour_repesentations, edge_weights)
        # Aggregate the neighbour messages.
        aggregated_messages = self.aggregate(
            node_indices, neighbour_messages, node_repesentations
        )
        # Update the node embedding with the neighbour messages.
        return self.update(node_repesentations, aggregated_messages)