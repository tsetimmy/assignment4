from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main(hidden_count, train_or_test):
    # Load training data
    f = open('../data/trainData.csv')
    train_data = np.loadtxt(f.readlines(), delimiter=',')
    f.close()

    f = open('../data/trainLabels.csv')
    train_labels = np.loadtxt(f.readlines()) - 5.0 # -5.0 to change to 0s and 1s
    train_labels = np.expand_dims(train_labels, axis=1)
    f.close()

    # Load testing data
    f = open('../data/' + train_or_test + 'Data.csv')
    test_data = np.loadtxt(f.readlines(), delimiter=',')
    f.close()

    f = open('../data/' + train_or_test + 'Labels.csv')
    test_labels = np.loadtxt(f.readlines()) - 5.0 # -5.0 to change to 0s and 1s
    f.close()

    # Parameters
    learning_rate = 0.001
    training_epochs = 1000 # Num of iterations in back propagation
    batch_size = len(train_data) # Don't divide into batches

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
        'output_b': tf.Variable(tf.random_uniform(shape=[n_output], minval=-0.5, maxval=0.5)),
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
        total_batch = int(len(train_data)/batch_size)
        for epoch in range(training_epochs):

            for i in range(total_batch):
                batch_xs = train_data
                _, c = sess.run([optimizer, cost], feed_dict={X: train_data, y_true: train_labels})
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

        probabilites = sess.run(
            y_pred, feed_dict={X: test_data})

        correct_count = 0
        assert len(probabilites) == len(test_labels)
        for i in range(len(probabilites)):
            if (probabilites[i][0] > 0.5 and
                test_labels[i] > 0.5 or
                probabilites[i][0] <= 0.5 and
                test_labels[i] <= 0.5):
                correct_count += 1
        print("Accuracy:", float(correct_count) / float(len(test_labels)))
        return float(correct_count) / float(len(test_labels))

if __name__ == '__main__':
    x = []

    train_accuracies = []
    for i in range(5, 16):
        print(i)
        x.append(i)
        acc = main(i, 'train')
        train_accuracies.append(acc)

    test_accuracies = []
    for i in range(5, 16):
        print(i)
        acc = main(i, 'test')
        test_accuracies.append(acc)

    xmin = 1.5
    assert len(train_accuracies) == len(test_accuracies)
    for i in range(len(train_accuracies)):
        xmin = min(xmin, train_accuracies[i])
        xmin = min(xmin, test_accuracies[i])

    ax = plt.subplot(121)
    plt.scatter(x, train_accuracies)
    ax.set_xlabel('Number of Hidden Nodes')
    ax.set_ylabel('Train Accuracy')
    ax.set_ylim([xmin - 0.005,1.005])
    plt.title('(a)')

    ax = plt.subplot(122)
    plt.scatter(x, test_accuracies)
    ax.set_xlabel('Number of Hidden Nodes')
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim([xmin - 0.005,1.005])
    plt.title('(b)')

    plt.tight_layout()
    #plt.show()
    plt.savefig('accuracy_nodes.pdf')
