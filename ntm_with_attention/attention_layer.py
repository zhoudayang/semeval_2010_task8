# coding=utf-8
import tensorflow as tf
from keras.layers import Layer
from keras import initializers
from keras import regularizers


class AttentionLayer(Layer):
    def __init__(self, units, kernel_initializer="glorot_uniform", kernel_regularizer=None):
        super(AttentionLayer, self).__init__()
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.units = units

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.score_units = input_shape[-1]
        self.times = input_shape[-2]
        self.score_kernel = self.add_weight(shape=(self.score_units, self.score_units),
                                            initializer=self.kernel_initializer,
                                            regularizer=self.kernel_regularizer,
                                            name="score_kernel")
        self.kernel = self.add_weight(shape=(self.units, self.score_units), initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      name="kernel")

        self.built = True

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # (None, 400)
        h_last = inputs[:, -1, :]
        # (None, 96, 400)
        output = tf.tensordot(inputs, self.score_kernel, axes=((2,), (1,)))
        # (None, 96)
        output = tf.einsum("bij,bj->bi", output, h_last)
        # (None, 96)
        output = tf.nn.softmax(output, axis=-1)
        # (None, 400)
        output = tf.einsum("bij,bi->bj", inputs, output)
        output = tf.tensordot(self.kernel, output, axes=((1,), (1,)))
        # todo : check 先乘,再transpose
        # (None, 80)
        output = tf.transpose(output, perm=(1, 0))
        # output = tf.reshape(output, shape=(batch_size, self.units))
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 3
        output_shape = list(input_shape)[:-1]
        output_shape[-1] = self.units
        return tuple(output_shape)
