
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.layers import Layer


def dense_layer(input_layer, num_units, activation, dropout=0.5, l2_reg=None, batch_norm=True, kernel_initializer=None):
    """
    Creates a dense layer in a Keras model.
    
    Parameters:
        - input_layer (tensor): input tensor to the dense layer
        - num_units (int): number of units in the dense layer
        - activation (string): activation function to use in the dense layer
        - dropout (float): dropout rate to use in the dense layer (default is 0.5)
        - l2_reg (float): L2 regularization coefficient (default is None)
        - bn (bool): whether to use batch normalization (default is True)
        - kernel_initializer (string): kernel initializer to use (default is None)
    
    Returns:
        - tensor: output tensor of the dense layer
    """

    if l2_reg:
        l2_reg = keras.regularizers.l2(l2_reg)
    else:
        l2_reg = None

    # create the dense layer
    dense_layer = keras.layers.Dense(
        num_units, 
        activation=None, 
        use_bias=False,  
        kernel_initializer=kernel_initializer,
        bias_initializer='zeros', 
        kernel_regularizer=l2_reg, 
        bias_regularizer=None,
        activity_regularizer=None, 
        kernel_constraint=None, 
        bias_constraint=None
    )(input_layer)

    # apply batch normalization if specified
    if batch_norm:
        dense_layer = keras.layers.BatchNormalization()(dense_layer)
    
    # apply activation function
    dense_layer = keras.layers.Activation(activation)(dense_layer)
    
    # apply dropout if specified
    if dropout:
        dense_layer = keras.layers.Dropout(dropout)(dense_layer)
        
    return dense_layer


def convolutional(inputs, num_filters, kernel_size, padding='same', activation='relu', dropout=0.2, l2_reg=None, batch_norm=True, kernel_initializer=None):
    """
    Creates a convolutional layer in a Keras model.
    
    Parameters:
        - inputs (tensor): input tensor to the convolutional layer
        - num_filters (int): number of filters in the convolutional layer
        - kernel_size (int): kernel size to use in the convolutional layer
        - padding (string): padding mode to use in the convolutional layer (default is 'same')
        - activation (string): activation function to use in the convolutional layer (default is 'relu')
        - dropout (float): dropout rate to use in the convolutional layer (default is 0.2)
        - l2_reg (float): L2 regularization coefficient (default is None)
        - bn (bool): whether to use batch normalization (default is True)
        - kernel_initializer (string): kernel initializer to use (default is None)
    
    Returns:
        - tensor: output tensor of the convolutional layer
    """
    if l2_reg:
        l2_reg = keras.regularizers.l2(l2_reg)
    else:
        l2_reg = None

    # create the convolutional layer
    conv_layer = keras.layers.Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        strides=1,
        activation=None,
        use_bias=False,
        padding=padding,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=l2_reg, 
        bias_regularizer=None, 
        activity_regularizer=None,
        kernel_constraint=None, 
        bias_constraint=None,
    )(inputs)        
    
    # apply batch normalization if specified
    if bn:                      
        conv_layer = keras.layers.BatchNormalization()(conv_layer)
    
    # apply activation function
    conv_layer = keras.layers.Activation(activation)(conv_layer)
    
    # apply dropout if specified
    if dropout:
        conv_layer = keras.layers.Dropout(dropout)(conv_layer)
    return conv_layer


    
def create_residual_block(input_layer, filter_size, activation='relu', l2_reg=None):
    """
    Creates a residual block in a Keras model.
    
    Parameters:
        - input_layer (tensor): input tensor to the residual block
        - filter_size (int): filter size to use in the convolutional layers in the residual block
        - activation (string): activation function to use in the residual block (default is 'relu')
        - l2_reg (float): L2 regularization coefficient (default is None)
    
    Returns:
        - tensor: output tensor of the residual block
    """

    if l2_reg:
        l2_reg = keras.regularizers.l2(l2_reg)
    else:
        l2_reg = None

    num_filters = input_layer.shape.as_list()[-1]  

    # create the first convolutional layer in the residual block
    res_block = keras.layers.Conv1D(
        filters=num_filters,
        kernel_size=filter_size,
        strides=1,
        activation='relu',
        use_bias=False,
        padding='same',
        dilation_rate=1,
        kernel_initializer='he_normal',
        kernel_regularizer=l2_reg)(input_layer) 
    # apply batch normalization
    res_block = keras.layers.BatchNormalization()(res_block)
    
    # apply activation function
    res_block = keras.layers.Activation(activation)(res_block)
    
    # create the second convolutional layer in the residual block
    res_block = keras.layers.Conv1D(
        filters=num_filters,
        kernel_size=filter_size,
        strides=1,
        activation='relu',
        use_bias=False,
        padding='same',
        dilation_rate=1,
        kernel_initializer='he_normal',
        kernel_regularizer=l2_reg
    )(res_block) 
    
    # apply batch normalization
    res_block = keras.layers.BatchNormalization()(res_block)
    
    # add the input layer to the residual block
    res_block = keras.layers.add([input_layer, res_block])
    
    # apply activation function
    return keras.layers.Activation(activation)(res_block)
