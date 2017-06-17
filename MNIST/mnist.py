"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""


# %% Import data
from tensorflow.examples.tutorials.mnist import input_data

# %% Import dependencies
import tensorflow as tf


# %% Implementing the Regression
def main(_):
    # import data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))


    #y = tf.nn.softmax(tf.matmul(x, W) + b)
    y = tf.matmul(x, W) + b

    # Training

    y_ = tf.placeholder(tf.float32, [None, 10])
    # This was the rax formulation of cross entropy. Can be numerically instable
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    # So we use the following function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    #1st way of init
    # init = tf.global_variables_initializer()
    # sess = tf.Session()
    # sess.run(init)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Evalating our model
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    ###############################################
    ########### Save model to play    #############
    ###############################################

# %% Run session
if __name__ == '__main__':

    tf.app.run(main=main)
