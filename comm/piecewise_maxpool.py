# coding:utf-8
import tensorflow as tf
from keras.engine.topology import Layer


class piecewise_maxpool_layer(Layer):
    def __init__(self, filter_num, fixed_size, **kwargs):
        self.filter_num = filter_num
        self.fixed_size = fixed_size
        super(piecewise_maxpool_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(piecewise_maxpool_layer, self).build(input_shape)

    def max_pool_piece1(self, x):
        conv_output = x[0]
        e1 = x[1]
        piece = tf.slice(conv_output, [0, 0], [e1 + 1, self.filter_num])
        return tf.reduce_max(piece, reduction_indices=[0])

    def max_pool_piece2(self, x):
        conv_output = x[0]
        e1 = x[1]
        e2 = x[2]
        piece = tf.slice(conv_output, [e1 + 1, 0], [e2 - e1, self.filter_num])
        return tf.reduce_max(piece, reduction_indices=[0])

    def max_pool_piece3(self, x):
        conv_output = x[0]
        e2 = x[1]
        piece = tf.slice(conv_output, [e2 + 1, 0], [-e2 + self.fixed_size - 1, self.filter_num])
        return tf.reduce_max(piece, reduction_indices=[0])

    def call(self, inputs):
        assert (len(inputs) == 3)
        # [None, fixed_size, filter_num]
        conv_output = inputs[0]
        e1 = inputs[1]
        e2 = inputs[2]
        e1 = tf.squeeze(e1)
        e2 = tf.squeeze(e2)
        # [None, filter_num]
        conv_piece1 = tf.map_fn(self.max_pool_piece1, (conv_output, e1), dtype=tf.float32)
        conv_piece2 = tf.map_fn(self.max_pool_piece2, (conv_output, e1, e2), dtype=tf.float32)
        conv_piece3 = tf.map_fn(self.max_pool_piece3, (conv_output, e2), dtype=tf.float32)
        # [None, 3 * filter_num]
        return tf.concat([conv_piece1, conv_piece2, conv_piece3], 1)

    def compute_output_shape(self, input_shape):
        return None, self.filter_num * 3
