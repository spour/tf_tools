from tensorflow import keras
from . import custom_layers, utility_functions


def build_model(conv_filters=(32, 124), kernel_sizes=(19, 5), activation='relu', pool_sizes=(25, 4), input_shape=200, num_units=512, num_outputs=12):
    """
    Constructs a 1D convolutional neural network for multi-class classification.
    
    Parameters:
    conv_filters (tuple of ints): Number of filters for each convolutional layer.
    kernel_sizes (tuple of ints): Size of the kernel for each convolutional layer.
    activation (str): Activation function to use for the first convolutional layer. 
                     Must be one of 'relu' (the default) or 'tanh'.
    pool_sizes (tuple of ints): Size of the pooling window for each max pooling layer.
    input_shape (int): Shape of the input data.
    num_units (int): Number of units in the fully-connected layer.
    num_outputs (int): Number of output classes.
    
    Returns:
    A compiled Keras model.
    """
    inputs = keras.layers.Input(shape=(input_shape, 4))
    activation_fn = utility_functions.activation_fn(activation)

    if input_shape == 1000:
        multiplier = 2
    else:
        multiplier = 1
        
    # input layer
    inputs = keras.layers.Input(shape=(input_shape,4))

    # layers
    #This loop block applies two convolutional layers with batch normalization and max pooling to the input tensor.
    x = inputs # initialize the output of the loop with the input layer
    for i in range(2): # loop through two iterations
        x = custom_layers.convolutional(filters=conv_filters[i],
         kernel_size=kernel_sizes[i], 
         activation=activation_fn,
          use_bias=False, 
          kernel_regularizer=keras.regularizers.l2(1e-6), 
          batch_norm=True)(x)
        x = keras.layers.MaxPool1D(pool_size=pool_sizes[i], strides=pool_sizes[i], padding='same')(x)
    
    # flatten and fully-connected layer
    x = keras.layers.Flatten()(x)
    x = custom_layers.dense_layer(num_units, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-6), use_batch_norm=True)(x)
    x = keras.layers.Dropout(0.5)(x)


    # Output layer 
    logits = keras.layers.Dense(12, activation='linear', use_bias=True)(x)
    outputs = keras.layers.Activation('sigmoid')(logits)

    # compile model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


"""


"""
