import tensorflow
from . import custom_layers, utility_functions


def build_model(act_fn: str ='relu'):
    """
    Builds a 1D convolutional neural network model for binary classification for genomics like Kelley's Basset model.

    Parameters:
    - act_fn (str): Activation function to use in the layers. Can be 'relu' or 'tanh'. Default is 'relu'.
   
    Returns:
    - A Keras model.
    """

    # input layer
    inputs = tf.keras.layers.Input(shape=(600,4))
    
    act_fn = utility_functions.get_activation(act_fn)
    
    # layer 1
    x = custom_layers.convolutional(inputs,
                           num_filters=300, 
                           kernel_size=19,   # 192
                           padding='same', 
                           activation=activation, 
                           dropout=0.2,
                           l2_reg=1e-6, 
                           batch_norm=True)
    x = tf.keras.layers.MaxPool1D(pool_size=3)(x)

    # layer 2
    x = custom_layers.convolutional(x,
                           num_filters=200, 
                           kernel_size=11,  # 56
                           padding='valid', 
                           activation='relu', 
                           dropout=0.2,
                           l2_reg=1e-6, 
                           batch_norm=True)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)

    # layer 3
    x = custom_layers.convolutional(x,
                           num_filters=200, 
                           kernel_size=7,  # 56
                           padding='valid', 
                           activation='relu', 
                           dropout=0.2,
                           l2_reg=1e-6, 
                           batch_norm=True)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)

    # layer 4
    x = tf.keras.layers.Flatten()(x)
    x = custom_layers.convolutional.dense_layer(x, num_units=1000, activation='relu', dropout=0.5, l2_reg=1e-6, batch_norm=True)

    # layer 5
    x = custom_layers.dense_layer(x, num_units=1000, activation='relu', dropout=0.5, l2_reg=1e-6, batch_norm=True)

    # Output layer 
    logits = tf.keras.layers.Dense(164, activation='linear', use_bias=True)(x)
    outputs = tf.keras.layers.Activation('sigmoid')(logits)
        
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

