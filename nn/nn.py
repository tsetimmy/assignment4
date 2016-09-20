from __future__ import division, print_function, absolute_import

import matplotlib
matplotlib.use("TkAgg")
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load training data
f = open('../data/trainData.csv')
print(type(f))
f.close()
exit()

f = open('../data/trainLabels.csv')
train_labels = f.read().split('\n')[:-1]
f.close()

# Parameters
learning_rate = 0.001
training_epochs = 1000 # Num of iterations in back propagation
batch_size = 1 # Don't divide into batches

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
    'hidden_b': tf.Variable(tf.random_uniform(shape=[n_hidden], minval=-0.5, maxval=0.5)),
    'output_b': tf.Variable(tf.random_uniform(shape=[n_hidden, n_output], minval=-0.5, maxval=0.5)),
}

# Build layer
def layer(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['hidden_h']),
                                   biases['hidden_b']))
    return layer

# Build output
def output(x):
    out = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['output_h']),
                                   biases['output_b']))
    return out

# Construct model
y_pred = output(layer(X))
# Targets
y_true = tf.placeholder("float", [None, n_output])

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, y_true: some_true_value})
        print("Epoch:", '%04d' % (epoch+1),
              "cost=", "{:.9f}".format(c))

