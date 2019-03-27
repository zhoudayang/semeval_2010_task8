# coding:utf-8
import tensorflow as tf
from keras.layers import Layer


class LossLayer(Layer):
    def __init__(self, **kwargs):
        super(LossLayer, self).__init__(**kwargs)

    def calculate_distance(self, w_output, r_embedding):
        w_output = w_output - r_embedding
        return tf.linalg.norm(w_output, axis=1)

    def get_loss(self, w_list, y_index):
        y_index = tf.cast(y_index, dtype=tf.int32)
        plus_score = w_list[y_index]
        _, indices = tf.nn.top_k(w_list, k=2)
        minus_index = tf.cond(tf.equal(y_index, indices[0]), lambda: indices[1], lambda: indices[0])
        minus_score = w_list[minus_index]
        return 1.0 + plus_score - minus_score

    def call(self, inputs):
        assert len(inputs) == 3
        w_output = inputs[0]
        batch_size = tf.shape(w_output)[0]
        r_embedding = inputs[1]
        label = inputs[2]

        output = tf.tensordot(w_output, r_embedding, axes=((1,), (1,)))

        output = tf.nn.softmax(output, axis=1)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=label))

        self.add_loss(loss, inputs=inputs)

        ret_val = tf.argmin(output, axis=1)

        return tf.one_hot(ret_val, depth=19)
