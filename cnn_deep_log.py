import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from . import custom_layers, utility_functions

def custom_activation(x):
    """
        Applies the custom activation function to the input tensor. This activation function 
        is similar to the ReLU activation function, but it has a non-zero gradient for negative 
        input values. This can potentially allow the model to learn more complex patterns in 
        the data.

    The custom activation function is defined as:
        if x > 0:
            return log(1 + exp(x))
        else:
            return -log(1 + exp(-x))

    Parameters:
    x: Tensor. The input tensor.

    Returns:
    Tensor. The output tensor with the same shape as x, where each element
    has been transformed by the custom activation function.
    """
    
    return tf.where(tf.math.greater(x, 0), 
                    tf.math.log(1 + tf.exp(x)), 
                    -tf.math.log(1 + tf.exp(-x)))


def build_model(l2_norm=True, input_shape=(200,4)):

    def l2_reg(weight_mat):
    """
    Calculates the L2 regularization term for a weight matrix. 0.1 is a hyperparameter 
    that determines the strength of the regularization. This term is added to the loss 
    function of a model to encourage the weights to take on smaller values, which can help 
    reduce overfitting.
    
    Parameters:
    weight_matrix (tensor): A matrix of weights.
    
    Returns:
    float: The L2 regularization term as scalar.
    
    Example:
    >>> l2_reg(tf.ones([3, 3]))
    0.1
    """
        return 0.1 * K.sum(K.square(weight_mat))

    if l2_norm:
        l2_first = l2_reg
    else:
        l2_first = None

    # input layer
    inputs = keras.Input(shape=input_shape)

    # layer 1
    x = custom_layers.convolutional(filters=32,
                     kernel_size=19,
                     strides=1,
                     activation=None,
                     use_bias=False,
                     padding='same',
                     kernel_regularizer=kernel_reg, 
                     )(inputs)        
    x = layers.BatchNormalization()(x)
    x = layers.Lambda(custom_activation)(x)
    x = layers.Dropout(0.1)(x)

    # layer 2
    x = custom_layers.convolutional(filters=48, 
                     kernel_size=7,   #176
                     padding='same', 
                     activation='relu', 
                     dropout=0.2,
                     kernel_regularizer=kernel_reg, 
                     )(x)
    x = layers.MaxPool1D(pool_size=4)(x)

    # layer 3
    x = custom_layers.convolutional(filters=96, 
                     kernel_size=7,     # 44
                     padding='valid', 
                     activation='relu', 
                     dropout=0.3,
                     kernel_regularizer=kernel_reg, 
                     )(x)
    x = layers.MaxPool1D(pool_size=4)(x)

    # layer 4
    x = custom_layers.convolutional(filters=128, 
                     kernel_size=3,   # 9
                     padding='valid', 
                     activation='relu', 
                     dropout=0.4,
                     kernel_regularizer=kernel_reg, 
                     )(x)
    
    x = layers.MaxPool1D(pool_size=3,  # 3
                        strides=3, 
                        padding='same'
                        )(x)

    # layer 5
    x = layers.Flatten()(x)
    x = layers.Dense(units=512, activation='relu', 
                     dropout=0.5, kernel_regularizer=kernel_reg)(x)

    # Output layer
    logits = layers.Dense(12, activation='linear', bias=True)(x)
    outputs = layers.Activation('sigmoid')(logits)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

