#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:20:30 2018

@author: jannes
"""

import tensorflow as tf
import numpy as np

DECAY = .999

def init_weights(shape, dist='random_normal', normalized=True):
    """Initializes network weights.
    
    Args:
        shape: A tensor. Shape of the weights.
        dist: A str. Distribution at initialization, one of 'random_normal' or 
            'truncated_normal'.
        normalized: A boolean. Whether weights should be normalized.
        
    Returns:
        A tf.variable.
    """
    # Normalized if normalized set to True
    if normalized == True:
        denom = np.prod(shape[:-1])
        std = 1 / denom
    else:
        std = .1
    
    # Draw from random or truncated normal
    if dist == 'random_normal':
        weights = tf.random_normal(shape, stddev=std)
    elif dist == 'truncated_normal':
        weights = tf.truncated_normal(shape, stddev=0.1)
    
    return tf.Variable(weights)


def init_biases(shape):
    """Initialize biases. """
    biases = tf.constant(0., shape=shape)
    return tf.Variable(biases)


def conv2d(x, W):
    """2D convolution. """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, pool_dim):
    """Max pooling. """
    patch_height = pool_dim[0]
    patch_width = pool_dim[1]
    return tf.nn.max_pool(x, 
                          ksize=[1, patch_height, patch_width, 1], 
                          strides=[1, patch_height, patch_width, 1], 
                          padding='SAME')


def leaky_relu_layer(x_in, n_in, n_out, patch_dim, n_chans, n_samples,
                     weights_dist='random_normal', normalized_weights=True,
                     is_first_layer=False):
    """Creates a CNN layer using the leaky ReLU non-linearity.
    
    Args:
        x_in: A tensor. Input neurons.
        n_in: An int. Number of input feature maps/channels.
        n_out: An int. Number of output feature maps/channels.
        patch_dim: A list of length 2. Dimensions of the convolution patch.
        n_chans: An int. Number of channels in the target area.
        n_samples: An int. Number of samples per trial.
        weights_dist: A str. Init weights from random or truncated normal.
        normalized_weights: A boolean. Whether weights should be normalized.
        is_first_layer: A boolean. Whether layer is first layer.
        
    Returns:
        A tensor of max-pooled feature maps.
    """
    # Initialize weights and biasesaccuracy
    W_conv = init_weights([patch_dim[0], patch_dim[1], n_in, n_out],
                          dist=weights_dist,
                          normalized=normalized_weights)
    b_conv = init_biases([n_out])
    
    # Only first layer has to be reshaped into 4d tensor
    if is_first_layer == True:
        x_in = tf.reshape(x_in, [-1, n_chans, n_samples, 1])
        
    # Convolute and activate using Leaky ReLu
    h_conv = tf.nn.leaky_relu(conv2d(x_in, W_conv) + b_conv)
    
    # Return max_pooled layer(s)
    return max_pool(h_conv)


def leaky_relu_batch(x_in, n_in, n_out, patch_dim, pool_dim, training, n_chans, 
                     n_samples, weights_dist='random_normal', 
                     normalized_weights=True, is_first_layer=False, bn=True):
    """Applies batch normalization from tf.contrib.layers after the ReLu.
    
    Args:
        x_in: A tensor. Input neurons.
        n_in: An int. Number of input feature maps/channels.
        n_out: An int. Number of output feature maps/channels.
        patch_dim: A list of length 2. Dimensions of the convolution patch.
        pool_dim: A list of length 2. Dimensions of the pooling patch.
        training: A boolean. Indicates training (True) or test (False).
        n_chans: An int. Number of channels/electrodes.
        n_samples: An int. Number of samples in the data.
        weights_dist: A str. Init weights from random or truncated normal.
        normalized_weights: A boolean. Whether weights should be normalized.
        is_first_layer: A boolean. Whether layer is first layer or not.
        bn: A boolean. Indicating whether batch-norm. should be applied.
        
    Returns:
        maxp_bn_relu: A tensor of max-pooled feature maps.
        weights: The weights tensor.
    """
    # Reshape if first layer, init weights
    if is_first_layer == True:
        x_in = tf.reshape(x_in, [-1, n_chans, n_samples, 1])
    weights = init_weights([patch_dim[0], patch_dim[1], n_in, n_out],
                          dist=weights_dist,
                          normalized=normalized_weights)
    
    # Batch-nomalize layer output (after ReLU)
    cnn = conv2d(x_in, weights)
    cnn_relu = tf.nn.leaky_relu(cnn)
    if bn == True:
        cnn_bn_relu = tf.contrib.layers.batch_norm(
                cnn_relu,
                data_format='NHWC',
                center=True,
                scale=True,
                is_training=training,
                decay=DECAY)
        maxp_bn_relu = max_pool(cnn_bn_relu, pool_dim)
    else:
        maxp_bn_relu = max_pool(cnn_relu, pool_dim)
    
    return maxp_bn_relu, weights


def elu_batch(x_in, n_in, n_out, patch_dim, pool_dim, training, n_chans, 
              n_samples, weights_dist='random_normal', normalized_weights=True, 
              is_first_layer=False, bn=True):
    """Applies batch normalization from tf.contrib.layers after the ELU.
    
    Args:
        x_in: A tensor. Input neurons.
        n_in: An int. Number of input feature maps/channels.
        n_out: An int. Number of output feature maps/channels.
        patch_dim: A list of length 2. Dimensions of the convolution patch.
        pool_dim: A list of length 2. Dimensions of the pooling patch.
        training: A boolean. Indicates training (True) or test (False).
        n_chans: An int. Number of channels/electrodes.
        n_samples: An int. Number of samples in the data.
        weights_dist: A str. Init weights from random or truncated normal.
        normalized_weights: A boolean. Whether weights should be normalized.
        is_first_layer: A boolean. Whether layer is first layer or not.
        bn: A boolean. Indicating whether batch-norm. should be applied.
        
    Returns:
        maxp_bn_relu: A tensor of max-pooled feature maps.
        weights: The weights tensor.
    """
    # Reshape if first layer, init weights
    if is_first_layer == True:
        x_in = tf.reshape(x_in, [-1, n_chans, n_samples, 1])
    weights = init_weights([patch_dim[0], patch_dim[1], n_in, n_out],
                          dist=weights_dist,
                          normalized=normalized_weights)
    
    # Batch-nomalize layer output (after ReLU)
    cnn = conv2d(x_in, weights)
    cnn_elu = tf.nn.elu(cnn)
    if bn == True:
        cnn_bn_elu = tf.contrib.layers.batch_norm(
                cnn_elu,
                data_format='NHWC',
                center=True,
                scale=True,
                is_training=training,
                decay=DECAY,
                renorm=True)
        maxp_bn_elu = max_pool(cnn_bn_elu, pool_dim)
    else:
        maxp_bn_elu = max_pool(cnn_elu, pool_dim)
    
    return maxp_bn_elu, weights


def create_network(n_layers, x_in, n_in, n_out, patch_dim, pool_dim, training, 
                   n_chans, n_samples, weights_dist='random_normal', 
                   normalized_weights=True, nonlin='leaky_relu', 
                   bn=True, keep_prob=0.5):
    """Creates arbritray number of hidden layers.
    
    Args:
        n_layers: An int. Number of hidden layers in the network.
        x_in: A tensor. Input neurons.
        n_in: An int. Number of input feature maps/channels.
        n_out: An int. Number of output feature maps/channels.
        patch_dim: A list of length 2. Dimensions of the convolution patch.
        pool_dim: A list of length 2. Dimensions of the pooling patch.
        training: A boolean. Indicates training (True) or test (False).
        n_chans: An int. Number of channels/electrodes.
        n_samples: An int. Number of samples in the data.
        weights_dist: A str. Init weights from random or truncated normal.
        normalized_weights: A boolean. Whether weights should be normalized.
        nonlin: A str. One of 'leaky_relu' or 'elu'; non-linearity to use.
        bn: A boolean. Indicating whether batch-norm. should be applied.
        
    Returns
        curr_output: Output of the last layer.
        weights: A dict of weights, one key/value pair per layer.
    """
    curr_in = x_in
    weights = {}
    for i in range(n_layers):
        is_first_layer = True if i == 0 else False
        
        if nonlin=='leaky_relu':
            curr_output, curr_weights = leaky_relu_batch(
                    x_in=curr_in,
                    n_in=n_in[i], 
                    n_out=n_out[i],
                    patch_dim=patch_dim,
                    pool_dim=pool_dim,
                    training=training,
                    n_chans=n_chans,
                    n_samples=n_samples,
                    weights_dist=weights_dist,
                    normalized_weights=normalized_weights,
                    is_first_layer=is_first_layer)
        elif nonlin=='elu':
             curr_output, curr_weights = elu_batch(
                    x_in=curr_in,
                    n_in=n_in[i], 
                    n_out=n_out[i],
                    patch_dim=patch_dim,
                    pool_dim=pool_dim,
                    training=training,
                    n_chans=n_chans,
                    n_samples=n_samples,
                    weights_dist=weights_dist,
                    normalized_weights=normalized_weights,
                    is_first_layer=is_first_layer)
        else:
            raise ValueError('Non-linearity "' + nonlin + '" not supported.')
        
        # Output of current layer is input of next, weights added to dict
        curr_in = curr_output
        weights[i] = curr_weights
        
    return curr_output, weights


def fully_connected(x_in, bn, units, training, nonlin='leaky_relu', 
                    weights_dist='random_normal', normalized_weights=True):
    """Adds fully connected layer.
    
    Args:
        x_in: A tensor. Input layer.
        bn: A boolean. Indicating whether batch-norm. should be applied.
        units: An int. Number of output units.
        training: A boolean. Indicates training (True) or test (False).
        nonlin: A str. One of 'leaky_relu' or 'elu'; non-linearity to use.
        
    Returns:
        out: Fully-connected output layer.
        weights: Weights for the output layer.
    """
    # Fully-connected layer (BN)
    shape_in = x_in.get_shape().as_list()
    dim = shape_in[1] * shape_in[2] * shape_in[3]
    weights = init_weights([dim, units],
                          dist=weights_dist,
                          normalized=normalized_weights)
    flat = tf.reshape(x_in, [-1, dim])
    h_conv = tf.matmul(flat, weights)
    
    # Batch-normalize fully-connected layer
    if bn == True:
        
        # Manual batch-normalization
# =============================================================================
#         batch_mean, batch_var = tf.nn.moments(h_conv, [0])    
#         h_conv_hat = (h_conv - batch_mean) / tf.sqrt(batch_var + 1e-3)
#         scale = tf.Variable(tf.ones([units]))
#         beta = tf.Variable(tf.zeros([units]))
#         if nonlin == 'leaky_relu':
#             layer_bn = tf.nn.leaky_relu(h_conv_hat)
#         elif nonlin == 'elu':
#             layer_bn = tf.nn.elu(h_conv_hat)
#         else:
#             raise ValueError('Non-linearity "' + nonlin + '" not supported.')
#         out = scale * layer_bn + beta
# =============================================================================
        
        if nonlin == 'leaky_relu':
            layer_bn = tf.nn.leaky_relu(h_conv)
            out = tf.contrib.layers.batch_norm(
                    layer_bn,
                    data_format='NHWC',
                    center=True,
                    scale=True,
                    is_training=training,
                    decay=DECAY)
        elif nonlin == 'elu':
            layer_bn = tf.nn.elu(h_conv)
            out = tf.contrib.layers.batch_norm(
                    layer_bn,
                    data_format='NHWC',
                    center=True,
                    scale=True,
                    is_training=training,
                    decay=DECAY,
                    renorm=True)
        else:
            raise ValueError('Non-linearity "' + nonlin + '" not supported.')
    else:
        if nonlin == 'leaky_relu':
            out = tf.nn.leaky_relu(h_conv)
        elif nonlin == 'elu':
            out = tf.nn.elu(h_conv)
        else:
            raise ValueError('Non-linearity \'' + nonlin + '\' not supported.')
    
    # Return fully-connected, batch-normalized output layer + weights
    return out, weights


def l2_loss(weights_cnn, l2_regularization_penalty, y_, y_conv, name):
    """Implements L2 loss for an arbitrary number of weights.
    
    Args:
        weights: A dict. One key/value pair per layer in the network.
        l2_regularization_penalty: An int. Scales the l2 loss arbitrarily.
        y_:
        y_conv:
        name: 
            
    Returns:
        L2 loss.        
    """
    weights = {}
    for key, value in weights_cnn.items():
        weights[key] = tf.nn.l2_loss(value)
    
    l2_loss = l2_regularization_penalty * sum(weights.values())
    
    unregularized_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    return tf.add(unregularized_loss, l2_loss, name=name)

def recall_macro(y_true, y_pred):
    ''' 
    Returns the recall macro (average of accuracy of each class) and its error bar.
    
    Args :
        y_true : Ndarray. Ground truth (correct) labels.
        y_pred : Predicted labels, as returned by a classifier.
    '''
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    n_test = np.sum(confusion_matrix, axis = 1)
    
        
    #fp = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
    fn = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    tp = np.diag(confusion_matrix)
    #tn = confusion_matrix.sum() - (fp + fn + tp)
    
    #
    recall_macro_per_class = tp/(tp+fn)
    #print(recall_macro_per_class)
    #
    recall_macro = np.mean(recall_macro_per_class)
    
    # l'écart-type théorique du recall macro, soit un interval de confiance de 0.68 = erf( 1/np.sqrt(2)) --> 1 fois l'écart-type
    error_bar = np.sqrt( np.sum(recall_macro_per_class * (1 - recall_macro_per_class)/n_test) ) /classes 
    
    return(recall_macro, error_bar)




