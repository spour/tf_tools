from tensorflow import keras
from . import custom_layers, utility_functions


def build_model(num_filters=24, kernel_size=19, 
activation='relu', dropout_rate=0.1, l2_reg=1e-6, 
use_batch_norm=True):
    """
    Constructs a 1D convolutional neural network for binary classification.
    
    Parameters:
    num_filters (int): Number of filters in the first convolutional layer.
    kernel_size (int): Size of the kernel in the first convolutional layer.
    activation (str): Activation function to use for the first convolutional layer. 
                     Must be one of 'relu' (the default) or 'tanh'.
    dropout_rate (float): Dropout rate for the first convolutional layer.
    l2_reg (float): L2 regularization strength for all layers.
    use_batch_norm (bool): Whether to use batch normalization in the first convolutional layer.
    
    Returns:
    A compiled Keras model.
    """
        
    # input layer
    inputs = keras.layers.Input(shape=(200, 4))
    activation_fn = utils.activation_fn(activation)

    # layer 1
    nn = custom_layers.convolutional(num_filters, kernel_size, activation=activation_fn, use_bias=False, kernel_regularizer=keras.regularizers.l2(l2_reg), batch_norm=use_batch_norm)(inputs)
    nn = keras.layers.Dropout(dropout_rate)(nn)
    nn = keras.layers.MaxPool1D(pool_size=50)(nn)

    # layer 2
    nn = custom_layers.convolutional(num_filters * 2, 3, activation='relu', use_bias=False, kernel_regularizer=keras.regularizers.l2(l2_reg), batch_norm=True)(nn)
    nn = keras.layers.Dropout(0.2)(nn)
    nn = keras.layers.MaxPool1D(pool_size=2)(nn)

    # layer 3
    nn = keras.layers.Flatten()(nn)
    nn = custom_layers.dense_layer(96, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg), use_batch_norm=True)(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    # Output layer 
    logits = keras.layers.Dense(1, activation='linear', use_bias=True)(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)

    # compile model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
