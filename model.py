import tensorflow as tf
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten

def conv_net(x):
    net = Conv2D(32, [7, 7], activation=tf.nn.relu, padding='same',
             data_format="channels_last", use_bias=True, 
             kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    
    net = Maxpooling2D([2, 2], padding='same')(net)
    
    net = Conv2D(64, [5, 5], activation=tf.nn.relu, padding='same',
             data_format="channels_last", use_bias=True, 
             kernel_initializer='glorot_uniform', bias_initializer='zeros')(net)
    
    net = Maxpooling2D([2, 2], padding='same')(net)
    
    net = Conv2D(128, [3, 3], activation=tf.nn.relu, padding='same',
             data_format="channels_last", use_bias=True, 
             kernel_initializer='glorot_uniform', bias_initializer='zeros')(net)
    
    net = Maxpooling2D([2, 2], padding='same')(net)
    
    net = Conv2D(256, [1, 1], activation=tf.nn.relu, padding='same',
             data_format="channels_last", use_bias=True, 
             kernel_initializer='glorot_uniform', bias_initializer='zeros')(net)
    
    net = Maxpooling2D([2, 2], padding='same')(net)
    
    net = Conv2D(28, [1, 1], activation=None, padding='same',
             data_format='channels_last', use_bias=True, 
             kernel_initializer='glorot_uniform', bias_initializer='zeros')(net)
    
    net = Maxpooling2D([2, 2], padding='same')(net)
    
    net = Flatten()(x)
    
    return net

def triplet_loss(model_anchor, model_positive, model_negative, margin):
        distance1 = tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(model_anchor - model_positive, 2), 1, keepdims=True))
        distance2 = tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(model_anchor - model_negative, 2), 1, keepdims=True))
        return tf.math.reduce_mean(tf.math.maximum(distance1 - distance2 + margin, 0))