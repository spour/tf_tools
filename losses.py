import tensorflow as tf

import tensorflow as tf

def binary_cross_entropy(true_labels, predicted_probabilities, weight=None, keep_dims=False, axis=None):
    """
    Calculate the binary cross entropy loss between true labels and predicted probabilities.
    :param true_labels: A tensor of shape (batch_size, num_classes) representing the true labels.
    :param predicted_probabilities: A tensor of shape (batch_size, num_classes) representing the predicted probabilities.
    :param weight: A tensor of shape (batch_size, num_classes) representing the weights for each sample in the batch.
    :param keep_dims: A boolean indicating whether to keep the dimensions or reduce to a scalar.
    :param axis: An integer or list of integers representing the dimensions to reduce.
    :return: A tensor representing the binary cross entropy loss.
    """
    # Clipping the predicted probabilities to prevent numeric instability
    predicted_probabilities = tf.clip_by_value(predicted_probabilities, 1e-7, 1 - 1e-7)
    if weight is not None:
        return tf.math.reduce_mean(weight * (true_labels * tf.math.log(predicted_probabilities) + (1 - true_labels) * tf.math.log(1 - predicted_probabilities)), keepdims=keep_dims, axis=axis)
    else:
        return tf.math.reduce_mean(true_labels * tf.math.log(predicted_probabilities) + (1 - true_labels) * tf.math.log(1 - predicted_probabilities



def categorical_cross_entropy(actuals: tf.Tensor, predictions: tf.Tensor, sample_weight: tf.Tensor = None, keep_dims: bool = False, axis: int = None):
    """
    Compute the categorical cross-entropy loss.
    :param actuals: The true labels.
    :param predictions: The predicted labels.
    :param sample_weight: The weight of each sample.
    :param keep_dims: Whether to keep the last dimension or not.
    :param axis: The axis along which to compute the loss.
    :return: The categorical cross-entropy loss.
    """
    predictions = tf.clip_by_value(predictions, 1e-7, 1 - 1e-7)
    if sample_weight is not None:
        return tf.math.reduce_mean(sample_weight * actuals * tf.math.log(predictions), keepdims=keep_dims, axis=axis)
    else:
        return tf.math.reduce_mean(actuals * tf.math.log(predictions), keepdims=keep_dims, axis=axis)
        

def squared_error_loss(actuals, predictions, sample_weights=None, keepdims=False, reduction_axis=None):
    """
    Calculates the squared error loss between actuals and predictions.
    :param actuals: The true values
    :param predictions: The predicted values
    :param sample_weights: Optional sample weights for the loss calculation
    :param keepdims: If true, the reduced dimensions are kept with size 1
    :param reduction_axis: The dimensions to reduce over. If None, reduces all dimensions
    :return: The squared error loss
    """
    if sample_weights:
        return tf.reduce_sum(sample_weights * tf.square(actuals - predictions), keepdims=keepdims, axis=reduction_axis)
    else:
        return tf.reduce_sum(tf.square(actuals - predictions), keepdims=keepdims, axis=reduction_axis)



def kullback_leibler_divergence(mu, log_var, axis=None):
    """
    Computes the Kullback-Leibler divergence between a Gaussian (mu, log_var) and a standard normal Gaussian.
    This is useful for Variational Autoencoders (VAEs).
    """
    sigma = tf.math.sqrt(tf.math.exp(log_var))
    return 0.5 * tf.math.reduce_sum(1 + 2 * tf.math.log(sigma) - tf.math.square(mu) - tf.math.exp(2 * tf.math.log(sigma)), axis=axis)


def kl_divergence_softmax(probabilities:tf.Tensor, axis:int=None):
    """
    This function calculates the KL divergence between a categorical distribution and a uniform distribution.
    It's commonly used in Variational Autoencoder (VAE) models.
    :param probabilities: a tensor of shape (batch_size, number_of_classes) representing the categorical distribution
    :param axis: the axis along which the reduction is performed. If not provided, the reduction is performed on all dimensions
    :return: a scalar tensor of KL divergence
    """
    num_classes = tf.shape(probabilities)[-1]
    # Clipping the input to avoid NaN from log(0)
    probabilities = tf.clip_by_value(probabilities,1e-7,1-1e-7)
    return tf.math.reduce_sum( probabilities*(tf.math.log(probabilities) - tf.math.log(1.0/num_classes)), axis=axis)
