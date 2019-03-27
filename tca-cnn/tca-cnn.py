# coding:utf-8
import numpy as np
from pretreat.semeval_2010_google import FIXED_SIZE
from pretreat.semeval_2010_google import EMBEDDING_DIM
from pretreat.semeval_2010_google import RELATION_COUNT, EMBEDDING_DIM
import os
import tensorflow as tf
import tensorlayer as tl
from layer import InputAttentionLayer, RelationEmbeddingLayer, TCAConv1dLayer, ScoringLayer
import time
from comm.helper import batch_iter


def score_cost(y, y_op, y_):
    batch_size = tf.shape(y)[0]
    row_indices = tf.range(batch_size, dtype=tf.int32)
    full_indices_op = tf.stack([row_indices, y_op], axis=1)
    full_indices_ = tf.stack([row_indices, y_], axis=1)
    # batch_size    for select one
    score_op = tf.gather_nd(y, full_indices_op)
    # batch_size    for ground truth
    score_ = tf.gather_nd(y, full_indices_)
    theta = score_op + 1.0 - score_
    result = tf.exp(theta + 1.0)
    result = tf.log(result)
    return result


if __name__ == "__main__":
    batch_size = 32
    POS_EMBEDDING_DIM = 80
    POS_EMBEDDING_VEC_LENGTH = 2 * FIXED_SIZE - 1
    RELATION_EMBEDDING_DIM = 100
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.chdir("/home/zy/data/zy_paper/new")
    train_relative_e1_pos = np.load("train_relative_e1_pos.npy")
    train_relative_e2_pos = np.load("train_relative_e2_pos.npy")
    test_relative_e1_pos = np.load("test_relative_e1_pos.npy")
    test_relative_e2_pos = np.load("test_relative_e2_pos.npy")
    train_labels = np.load("train_labels.npy")
    test_labels = np.load("test_labels.npy")
    vec = np.load("word_vec.npy")
    train_index = np.load("train_word_index.npy")
    test_index = np.load("test_word_index.npy")

    word_index_input = tf.placeholder(tf.int32, shape=[batch_size, FIXED_SIZE])
    relative_e1_pos_input = tf.placeholder(tf.int32, shape=[batch_size, FIXED_SIZE])
    relative_e2_pos_input = tf.placeholder(tf.int32, shape=[batch_size, FIXED_SIZE])
    # 这里输入的label不是one-hot类型的
    y_ = tf.placeholder(tf.int32, shape=[batch_size, RELATION_COUNT])
    y_output = tf.argmax(y_, axis=1, output_type=tf.int32)

    word_embedding_output = tl.layers.EmbeddingInputlayer(inputs=word_index_input,
                                                          vocabulary_size=len(vec),
                                                          embedding_size=EMBEDDING_DIM,
                                                          name="word_embedding_layer")
    relative_e1_embedding_output = tl.layers.EmbeddingInputlayer(inputs=relative_e1_pos_input,
                                                                 vocabulary_size=1000,
                                                                 embedding_size=POS_EMBEDDING_DIM,
                                                                 name="relative_e1_embedding_layer")
    relative_e2_embedding_output = tl.layers.EmbeddingInputlayer(inputs=relative_e2_pos_input,
                                                                 vocabulary_size=1000,
                                                                 embedding_size=POS_EMBEDDING_DIM,
                                                                 name="relative_e2_embedding_layer")

    R = RelationEmbeddingLayer(relation_cnt=RELATION_COUNT, relation_embedding_dim=RELATION_EMBEDDING_DIM)

    # None, FIXED_SIZE, WORD_EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM
    sentence_embedding_output = tl.layers.ConcatLayer(
        prev_layer=[word_embedding_output, relative_e1_embedding_output, relative_e2_embedding_output], concat_dim=-1,
        name="concat_layer")

    sentence_representation = InputAttentionLayer(sentence_embedding_output, R)
    conv_output = TCAConv1dLayer(sentence_representation, act=tf.nn.tanh,
                                 shape=(4, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM, 1000), stride=1, padding="SAME")

    score_output = ScoringLayer([R, conv_output])
    # 代表网络的输出
    y = score_output.outputs
    # 代表模型predict的输出
    y_op = tf.argmax(y, axis=1, output_type=tf.int32)
    # todo : 添加正则项
    cost = score_cost(y, y_op=y_op, y_=y_output)
    correct_prediction = tf.equal(y_op, y_output)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_params = score_output.all_params
    train_op = tf.train.AdagradOptimizer(learning_rate=0.002).minimize(cost, var_list=train_params)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # tl.files.assign_params(sess, vec, word_embedding_output)
    score_output.print_params()
    score_output.print_layers()

    n_epoch = 100

    batches = batch_iter(list(zip(train_index, train_relative_e1_pos, train_relative_e2_pos, train_labels)), batch_size,
                         n_epoch)

    for batch in batches:
        x_index, x_e1_pos, x_e2_pos, y_label = zip(*batch)
        x_index = np.array(x_index)
        x_e1_pos = np.array(x_e1_pos)
        x_e2_pos = np.array(x_e2_pos)
        y_label = np.array(y_label)

        feed_dict = {
            word_index_input: x_index,
            relative_e1_pos_input: x_e1_pos,
            relative_e2_pos_input: x_e2_pos,
            y_: y_label
        }

        _, loss, accuracy = sess.run([train_op, cost, acc], feed_dict)


        print loss, accuracy
# for epoch in range(n_epoch):
#     start_time = time.time()
#         tl.iterate.minibatches()
