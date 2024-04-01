from main import *
sns.set_style(style='whitegrid',rc={'font.family': 'sans-serif','font.serif':'Times'})
sns.color_palette('hls', 8)



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
    X_delta_mu = np.empty((num_delta_mu, num_samples, 700))

    if data:
      X_delta_mu = np.empty((num_delta_mu, num_samples, data[0].shape[1]))

    if '1090' in model.name.split('_'):
        X_delta_mu = np.empty((num_delta_mu, num_samples, 701))
        X = add_1090(np.expand_dims(X, axis=-1))

    for i in tqdm(range(len(X))):
        index = int(Y[i] // delta_mu)
        if counts[index] >= num_samples: continue
        else:
            X_delta_mu[index, int(counts[index]), :] = X[i]
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
Evaluate the efficacy of distribution shifting
'''
def shift_distribution_test(X_train, Y_train, X_test, Y_test):
    X_train_jitter = jitter_pulses(X_train, 300)
    X_test_jitter = jitter_pulses(X_test, 300)
    three_layer_mlp = MLPModel(input_size=700, output_size=1, classification=False, name='3_layer_mlp')
    load_model_weights(three_layer_mlp)

    train(three_layer_mlp, X_train_jitter, Y_train, epochs=100, batch_size=64)
    linearity_plot(three_layer_mlp, (X_test_jitter, Y_test), num_samples=500, num_delta_mu=30)

    three_layer_mlp_2 = MLPModel(input_size=700, output_size=1, classification=False, name='3_layer_mlp')
    load_model_weights(three_layer_mlp)
    # load_model_weights(three_layer_mlp_2)

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
    # load_model_weights(model)
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
        autoencoder = Autoencoder(input_size=700, encoder_layer_sizes=[512, latent_size])

        train(autoencoder, X_train, X_train, epochs=100, batch_size=256)

        for x in X_test[:2]:
            plt.plot(x, label='Original Pulse')
            plt.plot(autoencoder(np.array([x]))[0], label='Reconstructed Pulse')
            plt.title('Pulse reconstruction: ' + autoencoder.name)
            plt.legend()
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

        train(autoencoder, X_train, [X_train, AT_train], epochs=1, batch_size=256)

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