import tensorflow as tf

def conv_block(x, model_width, kernel):
    # 1D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Convolution1D(model_width, kernel, padding='same', kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


class VGG:
    def __init__(self, length, num_channel, num_filters, problem_type='Regression', output_nums=1, dropout_rate=False):
        self.length = length
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.dropout_rate = dropout_rate

    def VGG11(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        # Block 1
        x = conv_block(inputs, self.num_filters * (2 ** 0), 3)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 2
        x = conv_block(x, self.num_filters * (2 ** 1), 3)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 3
        x = conv_block(x, self.num_filters * (2 ** 2), 3)
        x = conv_block(x, self.num_filters * (2 ** 2), 3)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 4
        x = conv_block(x, self.num_filters * (2 ** 3), 3)
        x = conv_block(x, self.num_filters * (2 ** 3), 3)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 5
        x = conv_block(x, self.num_filters * (2 ** 3), 3)
        x = conv_block(x, self.num_filters * (2 ** 3), 3)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Fully Connected (MLP) block
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate, name='dropout')(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        # Create model.
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg11_1d')

        return model

    def VGG13(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        # Block 1
        x = conv_block(inputs, self.num_filters * (2 ** 0), 3)
        x = conv_block(x, self.num_filters * (2 ** 0), 3)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 2
        x = conv_block(x, self.num_filters * (2 ** 1), 3)
        x = conv_block(x, self.num_filters * (2 ** 1), 3)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 3
        x = conv_block(x, self.num_filters * (2 ** 2), 3)
        x = conv_block(x, self.num_filters * (2 ** 2), 3)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 4
        x = conv_block(x, self.num_filters * (2 ** 3), 3)
        x = conv_block(x, self.num_filters * (2 ** 3), 3)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 5
        x = conv_block(x, self.num_filters * (2 ** 3), 3)
        x = conv_block(x, self.num_filters * (2 ** 3), 3)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Fully Connected (MLP) block
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate, name='dropout')(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        # Create model.
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg13_1d')

        return model

    def VGG16_small(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        # Block 1
        x = conv_block(inputs, 16, 3)
        x = conv_block(x, 16, 3)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 2
        x = conv_block(x, 16, 3)
        x = conv_block(x, 16, 3)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 3
        x = conv_block(x, 32, 3)
        x = conv_block(x, 32, 3)
        x = conv_block(x, 32, 3)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 4
        x = conv_block(x, 32, 5)
        x = conv_block(x, 32, 5)
        x = conv_block(x, 32, 5)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 5
        x = conv_block(x, 32, 5)
        x = conv_block(x, 32, 5)
        x = conv_block(x, 32, 5)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Fully Connected (MLP) block
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate, name='dropout')(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        # Create model.
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg16_small_1d')

        return model

    def VGG16_v2(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        # Block 1
        x = conv_block(inputs, self.num_filters * (2 ** 0), 3)
        x = conv_block(x, self.num_filters * (2 ** 0), 3)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 2
        x = conv_block(x, self.num_filters * (2 ** 1), 3)
        x = conv_block(x, self.num_filters * (2 ** 1), 3)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 3
        x = conv_block(x, self.num_filters * (2 ** 2), 3)
        x = conv_block(x, self.num_filters * (2 ** 2), 3)
        x = conv_block(x, self.num_filters * (2 ** 2), 1)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 4
        x = conv_block(x, self.num_filters * (2 ** 3), 3)
        x = conv_block(x, self.num_filters * (2 ** 3), 3)
        x = conv_block(x, self.num_filters * (2 ** 3), 1)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 5
        x = conv_block(x, self.num_filters * (2 ** 3), 3)
        x = conv_block(x, self.num_filters * (2 ** 3), 3)
        x = conv_block(x, self.num_filters * (2 ** 3), 1)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Fully Connected (MLP) block
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate, name='dropout')(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        # Create model.
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg16_v2_1d')

        return model

    def VGG19(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        # Block 1
        x = conv_block(inputs, self.num_filters * (2 ** 0), 3)
        x = conv_block(x, self.num_filters * (2 ** 0), 3)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 2
        x = conv_block(x, self.num_filters * (2 ** 1), 3)
        x = conv_block(x, self.num_filters * (2 ** 1), 3)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 3
        x = conv_block(x, self.num_filters * (2 ** 2), 3)
        x = conv_block(x, self.num_filters * (2 ** 2), 3)
        x = conv_block(x, self.num_filters * (2 ** 2), 3)
        x = conv_block(x, self.num_filters * (2 ** 2), 3)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 4
        x = conv_block(x, self.num_filters * (2 ** 3), 3)
        x = conv_block(x, self.num_filters * (2 ** 3), 3)
        x = conv_block(x, self.num_filters * (2 ** 3), 3)
        x = conv_block(x, self.num_filters * (2 ** 3), 3)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 5
        x = conv_block(x, self.num_filters * (2 ** 3), 3)
        x = conv_block(x, self.num_filters * (2 ** 3), 3)
        x = conv_block(x, self.num_filters * (2 ** 3), 3)
        x = conv_block(x, self.num_filters * (2 ** 3), 3)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Fully Connected (MLP) block
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate, name='dropout')(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        # Create model.
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg19_1d')

        return model
    
def build_vgg(length, name, width, num_channel=1, problem_type='Regression', output_nums=1):
    if name == 'vgg13': model = VGG(length, num_channel, width, problem_type=problem_type, output_nums=output_nums, dropout_rate=0.4).VGG13()
    if name == 'vgg16_small': model = VGG(length, num_channel, width, problem_type=problem_type, output_nums=output_nums, dropout_rate=0.4).VGG16_small()
    if name == 'vgg16': model = VGG(length, num_channel, width, problem_type=problem_type, output_nums=output_nums, dropout_rate=0.4).VGG16_v2()
    model.loss = tf.keras.losses.MeanAbsoluteError()
    model.optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=model.optimizer, loss=model.loss)
    model.build((None, length, 1))

    return model