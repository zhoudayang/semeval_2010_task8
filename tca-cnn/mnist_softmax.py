# coding:utf-8
import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
    # import data
    mnist = input_data.read_data_sets("/home/zy/MNIST_data/")
    # create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.int64, [None])

    # the raw formulation of cross-entropy
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)), reduction_indices=[1]))
    # can be numerically unstable
    # so here we use tf.losses.sparse_softmax_cross_entropy on the raw outputs of 'y', and then
    # average across the batch

    # real, prediction
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for _ in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, cost = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
        print cost
    # test trained model
    # 输出最大的那一个神经元对应一个类
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 输入x和y_输出预测的准确率
    print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/tmp/tensorflow/mnist/input_data",
                        help="Directory for storing input data")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
