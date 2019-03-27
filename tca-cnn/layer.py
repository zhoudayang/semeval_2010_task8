# coding:utf-8

import tensorflow as tf
from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig
from tensorlayer import logging
import tensorlayer as tl

"""
a=np.arange(12).reshape(3,4)
a
array([[ 0, 1, 2, 3],
  [ 4, 5, 6, 7],
  [ 8, 9, 10, 11]])
a.reshape(4,3)
array([[ 0, 1, 2],
  [ 3, 4, 5],
  [ 6, 7, 8],
  [ 9, 10, 11]])
a.transpose(1,0)
array([[ 0, 4, 8],
  [ 1, 5, 9],
  [ 2, 6, 10],
  [ 3, 7, 11]])
之前对此一直有误解，两者有很大的区别，不能混用，一定要重视！！
"""


class RelationEmbeddingLayer(Layer):
    def __init__(self,
                 relation_cnt,
                 relation_embedding_dim,
                 R_init=tf.random_uniform_initializer(-0.1, 0.1),
                 R_init_args=None,
                 name="Rembedding"):
        super(RelationEmbeddingLayer, self).__init__(
            prev_layer=None,
            act=None,
            name=name,
            R_init_args=R_init_args
        )
        self.relation_cnt = relation_cnt
        self.relation_embedding_dim = relation_embedding_dim
        logging.info("RelationEmbeddingInputLayer %s:(%d, %d)" % (name, relation_cnt, relation_embedding_dim))
        with tf.variable_scope(name):
            R = tf.get_variable(name="r_embedding", shape=(relation_cnt, relation_embedding_dim), initializer=R_init,
                                dtype=LayersConfig.tf_dtype, **self.R_init_args)
            self.outputs = R
        self._add_layers(self.outputs)
        self._add_params(R)


class InputAttentionLayer(Layer):
    def __init__(self,
                 prev_layer,
                 R,
                 M_init=tf.truncated_normal_initializer(stddev=0.1),
                 b_init=tf.constant_initializer(value=0.0),
                 M_init_args=None,
                 b_init_args=None,
                 name="InputAttentionLayer",
                 ):
        super(InputAttentionLayer, self).__init__(
            prev_layer=prev_layer,
            act=None,
            name=name,
            M_init_args=M_init_args,
            b_init_args=b_init_args)
        relation_embedding_dim = R.relation_embedding_dim
        logging.info("AttentionInputLayer %s [%d]" % (self.name, relation_embedding_dim))
        self.relation_embedding_dim = relation_embedding_dim
        self.relation_cnt = R.relation_cnt
        if self.inputs.get_shape().ndims != 3:
            raise AssertionError("The input dimension must be rand 3[batch_size, sentence, word_representation]")

        word_representation_dim = int(self.inputs.get_shape()[-1])
        sentence_size = int(self.inputs.get_shape()[1])
        with tf.variable_scope(name):
            # todo : 根据实际效果进行调整
            M = tf.get_variable(name="M", shape=(word_representation_dim, relation_embedding_dim), initializer=M_init,
                                dtype=LayersConfig.tf_dtype, **self.M_init_args)
            bias = tf.get_variable(name="b", shape=(sentence_size,), initializer=b_init, dtype=LayersConfig.tf_dtype,
                                   **self.b_init_args)

            self.outputs = tf.tensordot(self.inputs, M, axes=((2), (0)), name="tensordot_1")
            self.outputs = tf.tensordot(self.outputs, R.outputs, axes=((2), (1)), name="tensordot_2")
            self.outputs = tf.transpose(self.outputs, perm=[0, 2, 1], name="transpose_1")
            self.outputs = tf.add(self.outputs, bias, name="add")
            self.outputs = tf.nn.softmax(self.outputs, name="softmax")
            # [None, 19, 100]
            self.outputs = tf.concat(values=[self.element_dot(self.outputs[:, i, :]) for i in range(self.relation_cnt)],
                                     axis=-1,
                                     name="concat")
            # [None, 100, 460, 19]
        self._add_layers(self.outputs)
        self._add_params([M, bias])

    def element_dot(self, input_tensor):
        input_tensor = tf.expand_dims(input=input_tensor, axis=-1)
        input_tensor = tf.multiply(self.inputs, input_tensor)
        input_tensor = tf.expand_dims(input=input_tensor, axis=-1)
        return input_tensor


class IndexLayer(Layer):
    def __init__(self,
                 inputs,
                 i,
                 name="IndexLayer"
                 ):
        super(IndexLayer, self).__init__(prev_layer=None, name=name)
        self.inputs = inputs
        self.index = i
        with tf.variable_scope(name):
            self.outputs = self.inputs[:, :, :, i]
        self._add_layers(self.outputs)


# 输入一维卷积的数据格式为: [batch, in_width, in_channels]
# filter的格式为 [filter_width, in_channels, out_channels],
# in_channels表示value一共有多少列，out_channels表示输出通道，表示移动有多少卷积核

class TCAConv1dLayer(Layer):
    def __init__(self,
                 prev_layer,
                 act=None,
                 shape=(5, 1, 5),
                 stride=1,
                 padding="SAME",
                 W_init=tf.truncated_normal_initializer(stddev=0.02),
                 b_init=tf.constant_initializer(value=0.0),
                 W_init_args=None,
                 b_init_args=None,
                 name="TCAConv1dLayer"
                 ):
        super(TCAConv1dLayer, self).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args,
                                             b_init_args=b_init_args, name=name)

        logging.info(
            "TCACConv1dLayer %s: shape: %s stride: %s pad: %s act: %s" % (
                self.name, str(shape), str(stride), padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

        if self.inputs.get_shape().ndims != 4:
            raise AssertionError(
                "The input dimension must be rand 3[batch_size, sentence, word_representation, relation_cnt]")
        self.relation_cnt = int(self.inputs.get_shape()[-1])
        self.sentence_size = int(self.inputs.get_shape()[1])
        with tf.variable_scope(name):
            self.index_layer = [IndexLayer(self.inputs, i, "IndexLayer_%d" % (i)) for i in range(self.relation_cnt)]
            self.conv_layer = [
                tl.layers.Conv1dLayer(layer, act=act, shape=shape, stride=stride, padding=padding, W_init=W_init,
                                      b_init=b_init, W_init_args=W_init_args, b_init_args=b_init_args,
                                      name="conv1d_%d" % (layer.index)) for layer in
                self.index_layer]
            self.maxpooling_layer = \
                [
                    tl.layers.MaxPool1d(layer, filter_size=self.sentence_size, strides=self.sentence_size,
                                        padding="valid", name="maxpooling1d_%d" % (i))
                    for i, layer in enumerate(self.conv_layer)
                ]
            self.outputs = tf.concat(
                [tf.expand_dims(input=tf.squeeze(layer.outputs, axis=1), axis=-1) for layer in self.maxpooling_layer],
                axis=-1)
            # (32, 1000, 19)
        self._add_layers(self.outputs)


class ScoringLayer(Layer):
    def __init__(self,
                 prev_layer,
                 U_init=tf.truncated_normal_initializer(stddev=0.1),
                 U_init_args=None,
                 name="ScoringLayer"
                 ):
        super(ScoringLayer, self).__init__(
            prev_layer=prev_layer,
            act=None,
            name=name,
            U_init_args=U_init_args
        )
        if len(self.inputs) != 2:
            raise AssertionError("the input layer's num must be 2")
        # 19 * 80
        self.R_output = self.inputs[0]
        # 32 * 1000 * 19
        self.conv_output = self.inputs[1]
        self.relation_cnt = int(self.R_output.get_shape()[0])
        self.relation_embedding_dim = int(self.R_output.get_shape()[-1])
        self.filter_num = int(self.conv_output.get_shape()[1])
        with tf.variable_scope(name):
            self.U = tf.get_variable(name="U", shape=(self.filter_num, self.relation_embedding_dim), initializer=U_init,
                                     dtype=LayersConfig.tf_dtype, **self.U_init_args)
            scores = tf.concat(
                [self.compute_score(self.conv_output[:, :, i], self.R_output[i, :]) for i in range(self.relation_cnt)],
                axis=1)
            self.outputs = scores
            # (32, 19)
        self._add_layers(self.outputs)
        self._add_params(self.U)

    def compute_score(self, conv_output, relation_embedding):
        conv_output = tf.expand_dims(input=conv_output, axis=1)
        relation_embedding = tf.expand_dims(input=relation_embedding, axis=-1)
        result = tf.tensordot(a=conv_output, b=self.U, axes=((2,), (0,)))
        result = tf.tensordot(a=result, b=relation_embedding, axes=((2,), (0,)))
        result = tf.squeeze(input=result, axis=-1)
        return result
