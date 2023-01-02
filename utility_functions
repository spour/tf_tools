import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow.compat.v1.keras.backend as K1


def make_directory(base_path: str, folder_name: str, verbose: int = 1) -> str:
    """Creates a directory at the specified base path with the given folder name.
    
    Parameters:
        - base_path (str): base path of the directory to create
        - folder_name (str): name of the folder to create
        - verbose (int): level of verbosity (default is 1)
        
    Returns:
        - str: full path of the created directory
    """
    # check if the base path exists and create it if necessary
    if not os.path.isdir(base_path):
        os.mkdir(base_path)
        if verbose:
            print(f"Making directory: {base_path}")

    # create the directory at the base path with the given folder name
    full_path = os.path.join(base_path, folder_name)
    if not os.path.isdir(full_path):
        os.mkdir(full_path)
        if verbose:
            print(f"Making directory: {full_path}")
    return full_path

    
def run_function_batch(sess, signed_grad, model, placeholders, inputs, batch_size: int = 128) -> np.ndarray:
    """Evaluates a function on a set of inputs, in batch form.
    
    Parameters:
        - sess (tf.Session): tensorflow session to use
        - signed_grad (Tensor): tensor to evaluate
        - model (Model): tensorflow model to use
        - placeholders (List[tf.Tensor]): list of tensorflow placeholders
        - inputs (List[np.ndarray]): list of numpy arrays as inputs
        - batch_size (int): size of each batch (default is 128)
        
    Returns:
        - np.ndarray: array of function evaluations
    """

    def feed_dict_batch(placeholders, inputs, index) -> Dict[tf.Tensor, np.ndarray]:
        """Creates a feed dictionary for a batch of inputs.
        
        Parameters:
            - placeholders (List[tf.Tensor]): list of tensorflow placeholders
            - inputs (List[np.ndarray]): list of numpy arrays as inputs
            - index (range): range of indices to use for this batch
            
        Returns:
            - Dict[tf.Tensor, np.ndarray]: feed dictionary for this batch
        """
        feed_dict = {}
        for i in range(len(placeholders)):
            feed_dict[placeholders[i]] = inputs[i][index]
        return feed_dict
    
    N = len(inputs[0])
    num_batches = int(np.floor(N/batch_size))
    
    values = []
    for i in range(num_batches):
        index = range(i*batch_size, (i+1)*batch_size)
        values.append(sess.run(signed_grad, feed_dict_batch(placeholders, inputs, index)))
    if num_batches*batch_size < N:
        index = range(num_batches*batch_size, N)
        values.append(sess.run(signed_grad, feed_dict_batch(placeholders, inputs, index)))
    values = np.concatenate(values, axis=0)

    return values


def calculate_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """Calculates class weights for training data.
    
    Parameters:
        - y_train (np.ndarray): array of training labels, with shape (num_samples, num_classes)
        
    Returns:
        - Dict[int, float]: dictionary of class weights, where keys are class indices and values are weights
    """
    count = np.sum(y_train, axis=0)
    weight = np.sqrt(np.max(count) / count)
    class_weights = {}
    for i in range(y_train.shape[1]):
        class_weights[i] = weight[i]
    return class_weights
    

def compile_regression_model(model: Model, learning_rate: float = 0.001, optimizer: str = 'adam', 
                             mask: bool = True, mask_val: Optional[Union[int, float]] = None, **kwargs) -> None:
    """Compiles a regression model.
    
    Parameters:
        - model (Model): tensorflow model to compile
        - learning_rate (float): learning rate to use (default is 0.001)
        - optimizer (str): optimizer to use (default is 'adam')
        - mask (bool): whether to apply a mask to the loss function (default is True)
        - mask_val (Union[int, float]): value to use for masking (default is None)
        - **kwargs: additional keyword arguments for the optimizer
        
    Returns:
        - None
    """
    optimizer = optimizers.get(optimizer)(learning_rate=learning_rate, **kwargs)
    if mask:
        def masked_loss_function(y_true, y_pred):
            mask = tf.cast(tf.math.logical_not(tf.math.equal(y_true, mask_val)), dtype=tf.float32)
            return keras.losses.mean_squared_error(y_true*mask, y_pred*mask)
        loss = masked_



def compile_classification_model(model: Model, loss_type: str = 'binary', optimizer: str = 'adam', 
                                 learning_rate: float = 0.001, monitor: List[str] = ['acc', 'auroc', 'aupr'], 
                                 label_smoothing: float = 0.0, from_logits: bool = False, **kwargs) -> None:
    """Compiles a classification model.
    
    Parameters:
        - model (Model): tensorflow model to compile
        - loss_type (str): type of loss function to use ('binary' or 'categorical', default is 'binary')
        - optimizer (str): optimizer to use (default is 'adam')
        - learning_rate (float): learning rate to use (default is 0.001)
        - monitor (List[str]): list of metrics to monitor (default is ['acc', 'auroc', 'aupr'])
        - label_smoothing (float): amount of label smoothing to apply (default is 0.0)
        - from_logits (bool): whether to apply the loss function to the logits (default is False)
        - **kwargs: additional keyword arguments for the optimizer
        
    Returns:
        - None
    """
    optimizer = optimizers.get(optimizer)(learning_rate=learning_rate, **kwargs)
    metrics = []
    if 'acc' in monitor:    
        metrics.append('accuracy')
    if 'auroc' in monitor:
        metrics.append(keras.metrics.AUC(curve='ROC', name='auroc'))
    if 'auroc' in monitor:
        metrics.append(keras.metrics.AUC(curve='PR', name='aupr'))

    if loss_type == 'binary':
        loss = keras.losses.BinaryCrossentropy(from_logits=from_logits, label_smoothing=label_smoothing)
    elif loss_type == 'categorical':
        loss = keras.losses.CategoricalCrossentropy(from_logits=from_logits, label_smoothing=label_smoothing)
    else:
        raise ValueError("Invalid loss type")

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
   


def get_optimizer(optimizer='adam', learning_rate=0.001, **kwargs):
    """Returns an optimizer given a name and optional hyperparameters.
    Parameters:
    optimizer : str, optional
        Name of the optimizer. Currently supports 'adam' and 'sgd'. Default is 'adam'.
    learning_rate : float, optional
        Learning rate for the optimizer. Default is 0.001.
    **kwargs : optional
        Additional arguments to pass to the optimizer.

    Returns:
    optimizer : tensorflow.keras.optimizers.*
        The optimizer object.
    """
    if optimizer == 'adam':
        beta_1 = kwargs.get('beta_1', 0.9)
        beta_2 = kwargs.get('beta_2', 0.999)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    elif optimizer == 'sgd':
        momentum = kwargs.get('momentum', 0.0)
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    else:
        raise ValueError(f"Unrecognized optimizer '{optimizer}'.")

    return optimizer



def clip_filters(filters, threshold=0.5, pad=3):
    """Clip filters with low entropy.
    Params:
        filters (List[np.ndarray]): List of filters. Each filter should be a 2D numpy array.
        threshold (float, optional): Threshold for minimum entropy. Filters with entropy below this value will be removed.
        pad (int, optional): Number of padding elements to add to either end of the filters being kept.

    Returns:
        List[np.ndarray]: List of clipped filters.

    Explain:
    The purpose of filtering out filters with low entropy is to remove filters that are less informative and might 
    be noise. These filters might be less useful for the task at hand, and removing them might improve the model's 
    performance. Clipping the filters with low entropy can help reduce the complexity of the model and make it 
    more interpretable. Additionally, removing filters with low entropy might also help reduce the risk of 
    overfitting by limiting the number of parameters in the model.
    """

    # Initialize list to hold the clipped filters
    filtered_filters = []

    # Iterate over each filter
    for w in filters:
        # Get the length and number of channels of the current filter
        length, channels = w.shape
        # Calculate the entropy of the filter
        entropy = np.log2(4) + np.sum(w*np.log2(w+1e-7), axis=1)
        # Get the indices of the filter where the entropy is above the threshold
        index = np.where(entropy > threshold)[0]
        # If there are any such indices
        if index.any():
            # Calculate the start index by taking the minimum of the indices minus the pad value, or 0 if that is negative
            start = max(np.min(index)-pad, 0)
            # Calculate the end index by taking the maximum of the indices plus the pad value plus 1, or the length of the filter if that is greater
            end = min(np.max(index)+pad+1, length)
            # Append the slice of the filter to the list of filtered filters
            filtered_filters.append(w[start:end,:])
        else:
            # If there are no indices above the threshold, append the entire filter to the list
            filtered_filters.append(w)

    # Return the list of filtered filters
    return filtered_filters



class ActivationFn:
    def activation_fn(self, activation):
        if activation == 'exp_relu':
            return self.exp_relu
        elif activation == 'shift_scale_tanh':
            return self.shift_scale_tanh
        elif activation == 'shift_scale_relu':
            return self.shift_scale_relu
        elif activation == 'shift_scale_sigmoid':
            return self.shift_scale_sigmoid
        elif activation == 'shift_relu':
            return self.shift_relu
        elif activation == 'shift_sigmoid':
            return self.shift_sigmoid
        elif activation == 'shift_tanh':
            return self.shift_tanh
        elif activation == 'scale_relu':
            return self.scale_relu
        elif activation == 'scale_sigmoid':
            return self.scale_sigmoid
        elif activation == 'scale_tanh':
            return self.scale_tanh
        elif activation == 'log_relu':
            return self.log_relu
        elif activation == 'log':
            return self.log
        elif activation == 'exp':
            return 'exponential'
        else:
            return activation

    def exp_relu(self, x, beta=0.001):
        return K.relu(K.exp(.1*x)-1)

    def log(self, x):
        return K.log(K.abs(x) + 1e-10)

    def log_relu(self, x):
        return K.relu(K.log(K.abs(x) + 1e-10))

    def shift_scale_tanh(self, x):
        return K.tanh(x-6.0)*500 + 500

    def shift_scale_sigmoid(self, x):
        return K.sigmoid(x-8.0)*4000

    def shift_scale_relu(self, x):
        return K.relu(K.pow(x-0.2, 3))

    def shift_tanh(self, x):
        return K.tanh(x-6.0)

    def shift_sigmoid(self, x):
        return K.sigmoid(x-8.0)

    def shift_relu(self, x):
        return K.relu(x-0.2)

    def scale_tanh(self, x):
        return K.tanh(x)*500 + 500

    def scale_sigmoid(self, x):
        return K.sigmoid(x)*4000

    def scale_relu(self, x):
        return K.relu((x)**3)
