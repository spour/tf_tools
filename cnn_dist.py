from tensorflow import keras
from . import custom_layers, utility_functions


def build_model(custom_activation='relu'):
    """
    Creates a CNN model for binary classification.
    
    Parameters
    ----------
    activation: str, optional
        Activation function to use for the layers. Default is 'relu'.
        
    Returns
    -------
    model: Keras model
        Compiled Keras model with the specified layers and activations.
    """
      
    # inputs
    inputs = keras.layers.Input(shape=(200,4))
    custom_activation = utility_functions.custom_activation_fn(custom_activation)

    # layer 1
    x = custom_layers.conv_layer(inputs,
                              num_filters=24, 
                              kernel_size=19, 
                              padding='same', 
                              activation=custom_activation, 
                              dropout=0.1,
                              l2_reg=1e-6, 
                              batch_norm=True)

    # layer 2
    x = custom_layers.conv_layer(x,
                              num_filters=32, 
                              kernel_size=7, 
                              padding='same', 
                              activation='relu', 
                              dropout=0.2,
                              l2_reg=1e-6, 
                              batch_norm=True)
    x = keras.layers.MaxPool1D(pool_size=4)(x)


    # layer 3
    x = custom_layers.conv_layer(x,
                              num_filters=48, 
                              kernel_size=7, 
                              padding='valid', 
                              activation='relu', 
                              dropout=0.3,
                              l2_reg=1e-6, 
                              batch_norm=True)
    x = keras.layers.MaxPool1D(pool_size=4)(x)

    # layer 4
    x = custom_layers.conv_layer(x,
                                num_filters=64, 
                                kernel_size=3, 
                                padding='valid', 
                                activation='relu', 
                                dropout=0.4,
                                l2_reg=1e-6, 
                                batch_norm=True)
    x = keras.layers.MaxPool1D(pool_size=3, 
                                strides=3, 
                                padding='same'
                                )(x)

    # layer 5
    x = keras.layers.Flatten()(x)
    x = custom_layers.dense_layer(x, num_units=96, activation='relu', 
                            dropout=0.5, l2_reg=1e-6, batch_norm=True)


    logits = keras.layers.Dense(1, activation='linear', bias=True)(x)
    outputs = keras.layers.Activation('sigmoid')(logits)
        
    #put together
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
