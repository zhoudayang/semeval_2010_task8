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
        # 测试loss层的输出结果
        ret_label = label
        # batch_size
        label = tf.argmax(label, axis=1)
        relation_cnt = tf.shape(r_embedding)[0]
        # 19 * batch_size
        result = tf.map_fn(lambda i: self.calculate_distance(w_output, r_embedding[i, :]), tf.range(relation_cnt),
                           dtype=tf.float32)
        # batch_size * 19
        result = tf.transpose(result, perm=(1, 0))
        ret_val = tf.argmax(result, axis=1)
        losses = tf.map_fn(lambda i: self.get_loss(result[i, :], label[i]), tf.range(batch_size), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
        self.add_loss(loss, inputs=inputs)
        return tf.one_hot(ret_val, depth=relation_cnt)
