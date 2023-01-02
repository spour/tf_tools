import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from . import custom_layers, utility_functions

def build_model(activation='log_relu', l2_norm=True):
    """
    Constructs a 1D convolutional neural network model for binary classification. 
    
    Parameters:
        - activation (str): Activation function to use for the convolutional layers. 
                           Must be one of 'log_relu' (the default) or 'relu'.
        - l2_norm (bool): Whether to apply L2 regularization to the model. Defaults to True.
    
    Returns:
        - A compiled Keras model.
    """
    
    def l2_reg(weight_matrix):
        """
        Calculates the L2 regularization term for a weight matrix.
        
        Parameters:
        weight_matrix (tensor): A matrix of weights.
        
        Returns:
        float: The L2 regularization term.
        """
        return 0.1 * K.sum(K.square(weight_matrix))


    l2 = 1e-6
    batch_norm = True

    dropout_0 = 0.1
    dropout_1 = 0.2
    dropout_2 = 0.3
    dropout_3 = 0.4
    dropout_4 = 0.5
         

    if l2_norm:
        l2_first = l2_reg
    else:
        l2_first = None
    # input layer
    inputs = keras.layers.Input(shape=(200,4))
    activation = utility_functions.activation_fn(activation)

    #  1
    x = keras.layers.Conv1D(filters=24,
                             kernel_size=19,
                             strides=1,
                             activation=None,
                             bias=False,
                             padding='same',
                             kernel_regularizer=l2_first, 
                             )(inputs)        
    x = keras.layers.BatchNormalization()(x)
    activation = utility_functions.activation_fn(activation)
    x = keras.layers.Activation(activation)(x)
    x = keras.layers.Dropout(dropout_0)(x)

    x = custom_layers.conv_layer(x,
                           num_filters=32, 
                           kernel_size=7, 
                           padding='same', 
                           activation='relu', 
                           dropout=dropout_1,
                           l2_reg=l2, 
                           batch_norm=batch_norm)
    x = keras.layers.MaxPool1D(pool_size=4, 
                                strides=4, 
                                padding='same'
                                )(x)

    x = custom_layers.conv_layer(x,
                           num_filters=48, 
                           kernel_size=7, 
                           padding='valid', 
                           activation='relu', 
                           dropout=dropout_2,
                           l2_reg=l2, 
                           batch_norm=batch_norm)
    x = keras.layers.MaxPool1D(pool_size=4, 
                                strides=4, 
                                padding='same'
                                )(x)

    # layer 2
    x = custom_layers.conv_layer(x,
                           num_filters=64, 
                           kernel_size=3, 
                           padding='valid', 
                           activation='relu', 
                           dropout=dropout_3,
                           l2_reg=l2, 
                           batch_norm=batch_norm)
    x = keras.layers.MaxPool1D(pool_size=3, 
                                strides=3, 
                                padding='same'
                                )(x)

    # Fully-coxected x
    x = keras.layers.Flatten()(x)
    x = custom_layers.dense_layer(x, num_units=96, activation='relu', dropout=dropout_block4, l2_reg=l2, batch_norm=batch_norm)

    # Output layer - additive + learned non-linearity
    logits = keras.layers.Dense(1, activation='linear', use_bias=True,  
                                 kernel_initializer='glorot_normal',
                                 bias_initializer='zeros')(x)
    outputs = keras.layers.Activation('sigmoid')(logits)
        
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
