from __future__ import division, print_function, absolute_import

import matplotlib
matplotlib.use("TkAgg")
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Parameters
learning_rate = 0.001
training_epochs = 1000 # Num of iterations in back propagation

hidden_count = 5
# Network Parameters
n_hidden = hidden_count # hidden layer num features (5 - 15)
n_input = 64
n_output = 1 # 1 sigmoid output

X = tf.placeholder("float", [None, n_input])

weights = {
    'hidden_h': tf.Variable(tf.random_uniform(shape=[n_input, n_hidden], minval=-0.5, maxval=0.5)),
    'output_h': tf.Variable(tf.random_uniform(shape=[n_hidden, n_output], minval=-0.5, maxval=0.5)),
}
biases = {
    'hidden_b': tf.Variable(tf.random_uniform(shape=[n_hidden], minval=-0.5, maxval=0.5))),
    'output_b': tf.Variable(tf.random_uniform(shape=[n_hidden, n_output], minval=-0.5, maxval=0.5)),
}

# Build layer
def layer(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['hidden_h']),
                                   biases['hidden_b']))
    return layer

# Build output
def output(x)
    out = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['output_h']),
                                   biases['output_b']))
    return out

# Construct model
op = output(layer(X))

