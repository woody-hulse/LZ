from main import *
sns.set_style(style='whitegrid',rc={'font.family': 'sans-serif','font.serif':'Times'})

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

    sns.set_theme(style="whitegrid")

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

    load_model_weights(three_layer_mlp)
    load_model_weights(three_layer_mlp_2)
    # load_model_weights(three_layer_mlp_1090)

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
        load_model_weights(model)
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
def plot_parameter_performance(path, lim=200, title='Number of Parameters vs. Training Performance'):
    parameters = []
    losses = []
    for dir in os.listdir(path):
        num_parameters = 0
        num_previous = 700
        if os.path.isfile(path + dir): continue
        with open(path + dir + '/trial.json') as f:
            trial = json.load(f)
            loss = trial['score']
            if loss > lim: continue
            for value in trial['hyperparameters']['values'].values():
                num_parameters += num_previous * value + value
                num_previous = value
            num_parameters += num_previous + 1
            parameters.append(num_parameters)
            losses.append(loss)
    
    plt.scatter(parameters, losses, lw=2)
    plt.xscale('log')
    plt.title(title)
    plt.show()