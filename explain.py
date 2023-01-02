import numpy as np
import pandas as pd
import logomaker
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow.compat.v1.keras.backend as K1
import shap



def saliency(model, X, class_index=0, layer=-2, batch_size=256):
    """
    Calculate the saliency score for the given input tensor and model.
    
    Parameters:
        - model (keras.Model): The model to use for saliency calculation.
        - X (numpy array): The input tensor to calculate saliency for.
        - class_index (int): The index of the class to calculate saliency for.
        - layer (int): The index of the layer to calculate saliency for.
        - batch_size (int): The batch size to use for saliency calculation.
        
    Returns:
        - A numpy array of saliency scores for each element in the input tensor.
    """

    # get the output tensor for the given layer
    output_tensor = model.layers[layer].output
    # calculate the gradient of the output tensor with respect to the input tensor
    saliency = K.gradients(output_tensor[:, class_index], model.input)[0]
    # get the current tensorflow session
    sess = K.get_session()

    # split the input tensor into batches
    num_elements = len(input_tensor)
    num_batches = int(np.floor(num_elements/batch_size))
    saliency_scores = []
    for i in range(num_batches):
        # calculate the saliency scores for each batch
        saliency_scores.append(sess.run(saliency, {model.input: input_tensor[i*batch_size:(i+1)*batch_size]}))
    if num_batches*batch_size < num_elements:
        # calculate the saliency scores for the remaining elements
        saliency_scores.append(sess.run(saliency, {model.input: input_tensor[num_batches*batch_size:num_elements]}))

    # concatenate the saliency scores for all batches into a single array
    return np.concatenate(saliency_scores, axis=0)


def mutagenesis(model, X, class_index=0, layer=-2):
    """
    Perform mutagenesis on the given input tensor and model to calculate the effect on the
    output of the model.
    
    Parameters:
        - model (keras.Model): The model to use for mutagenesis.
        - input_tensor (numpy array): The input tensor to perform mutagenesis on.
        - class_index (int): The index of the class to calculate mutagenesis for.
        - layer (int): The index of the layer to calculate mutagenesis for.
        
    Returns:
        - A numpy array of mutagenesis scores for each element in the input tensor.

    Explanation: 
        1. The function generates mutagenized versions of the input sequences, where 
        each position in the sequence is replaced with a one-hot encoded vector for 
        each possible alphabet character, and then calculates the difference between 
        the model's output for the wildtype sequence and the mutagenized sequences.
        2. The output is a 3D array of shape (batch size, sequence length, alphabet size)
        where each element represents the change in output caused by replacing the character 
        at that position with each possible alphabet character.
        3. To generate the mutagenized sequences, the function first defines a helper function 
        generate_mutagenesis which takes in an input tensor and returns a list of mutagenized 
        sequences where each position has been replaced with a one-hot encoded vector for 
        each alphabet character. This list is then converted into a NumPy array and returned.
        4. Next, the function creates an intermediate model by taking the inputs and outputs 
        of the original model, but replacing the output layer with the output of the specified 
        layer. The intermediate model is then used to predict the output for the wildtype 
        sequences and the mutagenized sequences. The difference between the wildtype score and 
        the mutagenized scores is then calculated and stored in a list, which is converted into 
        a NumPy array and returned.

    """

    def generate_mutagenesis(X):
        """
        Generate all possible mutagenized versions of the given input tensor.
        
        Parameters:
            - input_tensor (numpy array): The input tensor to perform mutagenesis on.
            
        Returns:
            - A numpy array of all possible mutagenized versions of the input tensor.
        """

        num_rows, num_cols = input_tensor.shape

        mutagenized_tensors = []
        for row in range(num_rows):
            for col in range(num_cols):
                new_tensor = np.copy(input_tensor)
                new_tensor[row,:] = 0
                new_tensor[row,col] = 1
                mutagenized_tensors.append(new_tensor)
        return np.array(mutagenized_tensors)

    num_samples, num_rows, num_cols = input_tensor.shape
    # create a new model that outputs the output of the specified layer 
    intermediate = keras.Model(inputs=model.inputs, outputs=model.layers[layer].output)

    # store mutagenesis scores for each sample
    mutagenesis_scores = []
    for sample in input_tensor:
        # get the baseline score for the wildtype sequence
        wildtype_score = intermediate_model.predict(np.expand_dims(sample, axis=0))[:, class_index]

        # generate mutagenized sequences
        mutagenized_sequences = generate_mutagenesis(sample)
        
        # get the scores for each mutagenized sequence
        mutagenized_scores = intermediate_model.predict(mutagenized_sequences)[:, class_index]

        # reshape the mutagenized scores into a 2D matrix
        mutagenesis_score = np.zeros((num_rows, num_cols))
        index = 0
        for row in range(num_rows):
            for col in range(num_cols):
                mutagenesis_score[row,col] = mutagenized_scores[index]
                index += 1
                
        # store the mutagenesis scores for this sample
        mutagenesis_scores.append(mutagenesis_score - wildtype_score)
    return np.array(mutagenesis_scores)


def deepshap(model, X, class_index=0, layer=-2, num_background=10, reference='shuffle'):
    """
    Compute SHAP values for the given model and sequences.
    
    Params:
        model: model used to calculate the SHAP values
        sequences: a numpy array of shape (num_sequences, sequence_length, alphabet_size)
        class_index: index of the class to compute SHAP values for. Default is 0.
        layer: layer of the model to compute SHAP values at. Default is -2.
        num_background: number of background sequences to use when reference is 'shuffle'. Default is 10.
        reference: reference method to use. Options are 'shuffle' or 'zero'. Default is 'shuffle'.
        If set to 'shuffle', the function will generate num_background number of shuffled versions 
        of each sequence and use these as the background dataset to calculate SHAP values. If set to 
        'zero', a background dataset of all zeros with the same shape as the input sequences will be used.

    Returns:
        attr_scores: a numpy array of shape (num_sequences, sequence_length) containing the SHAP values for each position. 

    Explain:
    1. The basic idea behind SHAP is to approximate the value of a function (the machine learning model) by 
    breaking it down into the sum of the values of its individual features. The value of each feature is calculated 
    using a cooperative game theory concept called Shapley values, which assigns a score to each player (feature) 
    based on their marginal contribution to the overall value of the game (the model output).
    2. calculation:
    2.i) Choose a reference dataset, which is used as a baseline for comparing the importance of the input features. 
    This can be either a dataset of all zeros or a dataset of randomly shuffled versions of the input data.
    2.ii) For each input feature, calculate the difference in the model output when the feature is present and when 
    it is absent (replaced with the corresponding value from the reference dataset). This difference is known as 
    the feature's "impact" on the model output.
    2.iii) Using the concept of Shapley values, assign a score to each feature based on its impact on the model output. 
    The final SHAP value for each feature is the sum of its individual Shapley value and the corresponding value from 
    the reference dataset.

    """
    num_sequences, sequence_length, alphabet_size = sequences.shape

    # if reference is not 'shuffle', set num_background to 1
    if reference != 'shuffle':
        num_background = 1
        
    # list to store SHAP values for each sequence
    shap_values = []

    # iterate through sequences and calculate SHAP values
    for i, seq in enumerate(sequences):
        if i % 50 == 0:
            print(f"{i+1} out of {num_sequences} sequences processed.")
        
        # generate background sequences
        if reference == 'shuffle':
            background_sequences = []
            for _ in range(num_background):
                shuffle = np.random.permutation(sequence_length)
                background_sequences.append(seq[shuffle, :])
            background_sequences = np.array(background_sequences)
        else: 
            background_sequences = np.zeros([1, sequence_length, alphabet_size])
        
        # add batch dimension to sequence and calculate SHAP values
        seq = np.expand_dims(seq, axis=0)
        explainer = shap.DeepExplainer(model, background_sequences)
        shap_values.append(explainer.shap_values(seq)[class_index])

    # concatenate SHAP values for all sequences
    attr_scores = np.concatenate(shap_values, axis=0)

    return attr_scores


 

def integrated_grad(model, X, class_index=0, layer=-2, num_background=10, num_steps=20, reference='shuffle'):

    def linear_path_sequences(x, num_background, num_steps, reference):
        def linear_interpolate(x, base, num_steps=20):
            x_interp = np.zeros(tuple([num_steps] +[i for i in x.shape]))
            for s in range(num_steps):
                x_interp[s] = base + (x - base)*(s*1.0/num_steps)
            return x_interp

        L, A = x.shape 
        seq = []
        for i in range(num_background):
            if reference == 'shuffle':
                shuffle = np.random.permutation(L)
                background = x[shuffle, :]
            else: 
                background = np.zeros(x.shape)        
            seq.append(linear_interpolate(x, background, num_steps))
        return np.concatenate(seq, axis=0)

    # setup op to get gradients from class-specific outputs to inputs
    saliency = K1.gradients(model.layers[layer].output[:,class_index], model.input)[0]

    # start session
    sess = K1.get_session()

    attr_score = []
    for x in X:
        # generate num_background reference sequences that follow linear path towards x in num_steps
        seq = linear_path_sequences(x, num_background, num_steps, reference)
       
        # average/"integrate" the saliencies along path -- average across different references
        attr_score.append([np.mean(sess.run(saliency, {model.inputs[0]: seq}), axis=0)])
    attr_score = np.concatenate(attr_score, axis=0)

    return attr_score


    
def attribution_score(model, X, method='saliency', norm='times_input', class_index=0,  layer=-2, **kwargs):
    """Compute attribution scores for the given model and sequences using the specified method.
    
    Params:
    model: model used to calculate the attribution scores
    sequences: a numpy array of shape (num_sequences, sequence_length, alphabet_size)
    method: method to use for calculating attribution scores. Options are 'saliency', 'mutagenesis', 'deepshap', and 
    'integrated_grad'. Default is 'saliency'.
    norm: normalization method to use. Options are 'l2norm' and 'times_input'. Default is 'times_input'.
    class_index: index of the class to compute attribution scores for. Default is 0.
    layer: layer of the model to compute attribution scores at. Default is -2.
    **kwargs: additional arguments to pass to the chosen method.

    Returns:
    attr_scores: a numpy array of shape (num_sequences, sequence_length, alphabet_size) containing the attribution 
    scores for each position in the sequence.
    """

    num_sequences, sequence_length, alphabet_size = sequences.shape

    # choose the method for calculating attribution scores based on the input argument
    if method == 'saliency':
        # get batch size from kwargs or use default value of 256
        batch_size = kwargs.get('batch_size', 256)
        # calculate attribution scores using the saliency method
        attr_scores = saliency(model, sequences, class_index, layer, batch_size)

    elif method == 'mutagenesis':
        # calculate attribution scores using the mutagenesis method
        attr_scores = mutagenesis(model, sequences, class_index, layer)
        
    elif method == 'deepshap':
        # get num_background and reference from kwargs or use default values
        num_background = kwargs.get('num_background', 5)
        reference = kwargs.get('reference', 'shuffle')
        # calculate attribution scores using the deepshap method
        attr_scores = deepshap(model, sequences, class_index, layer, num_background, reference)

    elif method == 'integrated_grad':
        # get num_background, num_steps, and reference from kwargs or use default values
        num_background = kwargs.get('num_background', 10)
        num_steps = kwargs.get('num_steps', 20)
        reference = kwargs.get('reference', 'shuffle')
        attr_scores = integrated_grad(model, sequences, class_index, layer, num_background, num_steps, reference)

    if norm == 'l2norm':
        # normalize the attribution scores using the chosen method
        # L2 normalization
        attr_scores = np.sqrt(np.sum(np.squeeze(attr_scores)**2, axis=2, keepdims=True) + 1e-10)
        attr_scores =  sequences * np.matmul(attr_scores, np.ones((1, alphabet_size)))
        
    elif norm == 'times_input':
        # element-wise multiplication of the attribution scores with the input sequences
        attr_scores *= sequences

    return attr_score


#-------------------------------------------------------------------------------------------------
# Plot conv filters
#-------------------------------------------------------------------------------------------------


def plot_filers(model, x_test, layer=3, threshold=0.5, window=20, num_cols=8, figsize=(30,5)):
    """Plots sequence logos for the filters of the given model at the specified layer.

    Params:
    model: model to extract filters from
    test_data: test data used to calculate activation patterns for the filters
    layer: layer of the model to extract filters from. Default is 3.
    threshold: threshold for filtering out weak activations. Default is 0.5.
    window: window size for calculating activation patterns. Default is 20.
    num_cols: number of columns to use in the plot. Default is 8.
    figsize: figure size for the plot. Default is (30,5).
    
    Returns:
    fig: matplotlib figure object
    W: list of filter weights
    logo: list of sequence logos

    """
    # create a model that outputs the activations of the specified layer
    intermediate_model = keras.Model(inputs=model.inputs, outputs=model.layers[layer].output)
    # get the activations for the test data
    activations = intermediate_model.predict(test_data)
    # calculate the activation patterns for the filters using the specified threshold and window size
    W = activation_pwm(activations, test_data, threshold=threshold, window=window)

    num_filters = len(W)
    num_widths = int(np.ceil(num_filters/num_cols))

    # create a figure with the specified figure size
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    logos = []
    for n, w in enumerate(W):
        ax = fig.add_subplot(num_widths, num_cols, n+1)

        # calculate sequence logo heights
        I = np.log2(4) + np.sum(w * np.log2(w+1e-10), axis=1, keepdims=True)
        logo = np.maximum(I*w, 1e-7)

        L, A = w.shape
        # create a dataframe with the logo counts
        counts_df = pd.DataFrame(data=0.0, columns=list('ACGT'), index=list(range(L)))
        for a in range(A):
            for l in range(L):
                counts_df.iloc[l,a] = logo[l,a]
        # create the sequence logo using the counts dataframe
        logomaker.Logo(counts_df, ax=ax)
        ax = plt.gca()
        ax.set_ylim(0,2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        plt.xticks([])
        plt.yticks([])

        logos.append(logo)
    
    return fig, W, logo


def calculate_activation_pwm(activations, sequences, threshold=0.5, window=20):
    """Calculates position probability matrices (PWMs) for the filters of the 
    given model based on activations and input sequences.

    Params:
    activations: activations of a model for a set of input sequences
    sequences: input sequences used to calculate the activations
    threshold: threshold for filtering out weak activations. Default is 0.5.
    window: window size for calculating the PWMs. Default is 20.
    
    Returns:
    W: list of PWMs for each filter
    """
    # determine window size for aligning sequences with activations
    window_left = int(window/2)
    window_right = window - window_left

    # get shape of input sequences
    num_sequences, sequence_length, alphabet_size = sequences.shape
    # get number of filters in the model
    num_filters = activations.shape[-1]

    W = []
    for filter_index in range(num_filters):
        # find activations above the given threshold
        coords = np.where(activations[:,:,filter_index] > np.max(activations[:,:,filter_index])*threshold)
        
        if len(coords) > 1:
            # get the indices of the strong activations
            x, y = coords
            # sort the activations by strength
            index = np.argsort(activations[x,y,filter_index])[::-1]
            data_index = x[index].astype(int)
            pos_index = y[index].astype(int)

            # create a sequence alignment centered around each strong activation
            seq_align = []
            for i in range(len(pos_index)):
                # determine the start and end positions of the window around the activation
                start_window = pos_index[i] - window_left
                end_window = pos_index[i] + window_right
                # make sure the window positions are valid
                if (start_window > 0) and (end_window < sequence_length):
                    seq = sequences[data_index[i], start_window:end_window, :]
                    seq_align.append(seq)

            # calculate the PWM for the filter
            if len(seq_align) > 1:
                W.append(np.mean(seq_align, axis=0))
            else: 
                W.append(np.ones((window,alphabet_size))/alphabet_size)
        else:
            # if there are no strong activations, create a uniform PWM
            W.append(np.ones((window,alphabet_size))/alphabet_size)

    return np.array(W)
