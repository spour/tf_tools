import tensorflow as tf
from . import custom_layers, utility_functions


def build_model(act_fn: str = 'relu', input_shape: int = 200, initialization: str = None):
    """
    Builds a 1D convolutional neural network model for binary classification.

    Parameters:
    - act_fn (str): Activation function to use in the layers. Can be 'relu' or 'tanh'. Default is 'relu'.
    - input_shape (int): Shape of the input tensor. Default is 200.
    - initialization (str): Initialization method for the weights of the layers. Can be 'glorot_uniform', 
                            'he_uniform', or None. If None, 'he_uniform' is used. Default is None.
   
    Returns:
    - A compiled Keras model.
    """

    if not initialization:
        initialization == 'he_uniform'

    if input_shape == 1000:
        multiplier = 2
    else:
        multiplier = 1     

    # input layer
    inputs = tf.keras.layers.Input(shape=(input_shape,4))
    
    # layer 1
    act_fn = utility_functions.get_activation(act_fn)
    x = custom_layers.convolutional(inputs,
                                   filters=32*multiplier, 
                                   kernel_size=19,  
                                   padding='same', 
                                   activation=act_fn, 
                                   kernel_initializer=initialization,
                                   dropout_rate=0.1,
                                   l2_reg=1e-6, 
                                   batch_norm=True)

    # layer 2
    x = custom_layers.convolutional(x,
                                   filters=48*multiplier, 
                                   kernel_size=7,   
                                   padding='same', 
                                   activation='relu', 
                                   dropout_rate=0.2,
                                   l2_reg=1e-6, 
                                   batch_norm=True)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)

    # layer 3
    x = custom_layers.convolutional(x,
                                   filters=96*multiplier, 
                                   kernel_size=7,     
                                   padding='valid', 
                                   activation='relu', 
                                   dropout_rate=0.3,
                                   l2_reg=1e-6, 
                                   batch_norm=True)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)

    # layer 4
    x = custom_layers.convolutional(x,
                                   filters=128*multiplier, 
                                   kernel_size=3,   
                                   padding='valid', 
                                   activation='relu', 
                                   dropout_rate=0.4,
                                   l2_reg=1e-6, 
                                   batch_norm=True)
    x = tf.keras.layers.MaxPool1D(pool_size=3)(x)

    # layer 5
    x = tf.keras.layers.Flatten()(x)
    x = custom_layers.dense(x, units=512*multiplier, activation='relu', 
                            dropout_rate=0.5, l2_reg=1e-6, batch_norm=True)

    # Output layer 
    logits = tf.keras.layers.Dense(12, activation='linear', bias=True)(x)
    outputs = tf.keras.layers.Activation('sigmoid')(logits)
        
    # compile model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model
