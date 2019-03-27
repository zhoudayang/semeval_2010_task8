# coding=utf-8
import sys

sys.path.append("..")

import numpy as np
import keras
import os
from keras.layers import Lambda, Embedding, LSTM, Dense, Input, Conv1D, concatenate, Conv2D, Activation, Dropout
from pretreat.semeval_2010_google import EMBEDDING_DIM, RELATION_COUNT, FIXED_SIZE
from keras.optimizers import SGD, Adam
import tensorflow as tf
from keras.callbacks import Callback
from comm.scorer import get_marco_f1
from keras.callbacks import EarlyStopping, ModelCheckpoint
from objective_function import LossLayer

# 视情况调整pos embedding的值
POS_EMBEDDING_DIM = 150
# word window size
K = 3
# 卷积核数目
CONV_SIZE = 1000
RELATION_EMBEDDING_DIM = CONV_SIZE


# 在开头和末尾添加padding
def add_padding(tensor):
    padding_length = (K - 1) / 2
    begin_tensor = tensor[:, :padding_length, :]
    end_tensor = tensor[:, -padding_length:, :]
    return tf.concat(values=[begin_tensor, tensor, end_tensor], axis=1)


# todo: 检查应该将什么作为channel？ 检查输入channel为1的情况
def sliding_func(tensor):
    batch_size = tf.shape(tensor)[0]
    fixed_size = tf.shape(tensor)[1]
    embedding_dim = tf.shape(tensor)[2]
    # [FIXED_SIZE, NONE, 3, EMBEDDING_DIM]
    tensor = tf.map_fn(lambda i: tensor[:, i:i + K, :], tf.range(FIXED_SIZE), dtype=tf.float32)
    # [NONE, FIXED_SIZE, 3, EMBEDDING_DIM]
    # return tf.transpose(tensor, perm=(1, 0, 2, 3))
    tensor = tf.transpose(tensor, perm=(1, 0, 3, 2))
    tensor = tf.reshape(tensor, shape=(batch_size, FIXED_SIZE, embedding_dim, K))
    return tensor


def tensordot(input_list):
    x = input_list[0]
    y = input_list[1]
    return tf.tensordot(x, y, axes=((1,), (1,)))


class f1_calculator(Callback):
    def __init__(self, index, e1_pos, e2_pos, relation_arr, result):
        self.index = index
        self.e1_pos = e1_pos
        self.e2_pos = e2_pos
        self.relation_arr = relation_arr
        self.result = result

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        predict_result = self.model.predict(x=[self.index, self.e1_pos, self.e2_pos, self.relation_arr, self.result])
        f1_score = get_marco_f1(predict_result, self.result)
        self.save_best(f1_score)
        logs["f1_score"] = f1_score

    def save_best(self, f1):
        with open("/home/zy/git/zy_paper/cnn_multi_attention/best/README", "r") as file:
            best_f1 = file.readline()
            best_f1 = float(best_f1)
        if f1 > best_f1:
            os.system("rm /home/zy/git/zy_paper/cnn_multi_attention/best/README")
            with open("/home/zy/git/zy_paper/cnn_multi_attention/best/README", "w") as file:
                file.write(str(f1))
            self.model.save("/home/zy/git/zy_paper/cnn_multi_attention/best/simple_cnn.model")


# todo: 实现without attention + without object function
# todo: check the difference between uniform and normal initializer

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    os.chdir("/home/zy/data/zy_paper/google")
    train_labels = np.load("train_labels.npy")
    test_labels = np.load("test_labels.npy")
    train_index = np.load("train_google_without_indicator_index.npy")
    test_index = np.load("test_google_without_indicator_index.npy")
    train_relative_e1_pos = np.load("train_relative_e1_pos_without_indicator.npy")
    train_relative_e2_pos = np.load("train_relative_e2_pos_without_indicator.npy")
    test_relative_e1_pos = np.load("test_relative_e1_pos_without_indicator.npy")
    test_relative_e2_pos = np.load("test_relative_e2_pos_without_indicator.npy")

    LENGTH = train_labels.shape[0] + test_labels.shape[0]

    TRAIN_LENGTH = train_labels.shape[0]
    TEST_LENGTH = LENGTH - LENGTH

    relation_input_arr = np.array([np.arange(19) for i in range(LENGTH)]).astype(np.int32)

    train_relation_input_arr = relation_input_arr[:TRAIN_LENGTH]
    test_relation_input_arr = relation_input_arr[TRAIN_LENGTH:]

    vec = np.load("google_vec.npy")

    relation_input_constants = list(range(0, RELATION_COUNT))
    K_constant = tf.constant(relation_input_constants, dtype=tf.int32)
    relation_input = Input(shape=(RELATION_COUNT,), dtype="int32")

    index_input = Input(shape=(FIXED_SIZE,), dtype="int32")
    relative_e1_index_input = Input(shape=(FIXED_SIZE,), dtype="int32")
    relative_e2_index_input = Input(shape=(FIXED_SIZE,), dtype="int32")

    label_input = Input(shape=(RELATION_COUNT,), dtype="int32")

    word_embedding_output = Embedding(input_dim=len(vec), output_dim=EMBEDDING_DIM, input_length=FIXED_SIZE,
                                      trainable=True)(index_input)
    # todo : check embedding_initializer as "normal"
    pos_embedding_output1 = Embedding(input_dim=2 * FIXED_SIZE + 1, output_dim=POS_EMBEDDING_DIM,
                                      input_length=FIXED_SIZE, trainable=True)(
        relative_e1_index_input)

    relation_embedding = Embedding(input_dim=RELATION_COUNT, output_dim=RELATION_EMBEDDING_DIM,
                                   input_length=RELATION_COUNT, trainable=True)

    pos_embedding_output2 = Embedding(input_dim=2 * FIXED_SIZE + 1, output_dim=POS_EMBEDDING_DIM,
                                      input_length=FIXED_SIZE, trainable=True)(
        relative_e2_index_input)
    # [None, FIXED_SIZE, EMBEDDING_DIM + POS_EMBEDDING_DIM + 2]
    sentence_embedding_output = concatenate([word_embedding_output, pos_embedding_output1, pos_embedding_output2],
                                            axis=2)

    # (None, FIXED_SIZE + 2, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM)
    padding_layer = Lambda(lambda x: add_padding(x))
    sentence_embedding_output = padding_layer(sentence_embedding_output)

    sliding_window_layer = Lambda(sliding_func)

    # print sliding_window_layer.compute_output_shape((None, FIXED_SIZE + 2, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM))

    Z = sliding_window_layer(sentence_embedding_output)

    conv2_layer = Conv2D(filters=CONV_SIZE, kernel_size=(1, 2 * POS_EMBEDDING_DIM + EMBEDDING_DIM), strides=(1, K),
                         padding="valid", activation="tanh")

    cnn_output = conv2_layer(Z)

    cnn_output = Lambda(lambda x: tf.squeeze(x, axis=2))(cnn_output)

    # print conv2_layer.compute_output_shape((None, FIXED_SIZE, 3, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM))
    # None, FIXED_SIZE, 1000
    # cnn_layer = Conv1D(filters=1000, kernel_size=K, strides=1, padding="valid", activation="tanh")

    # cnn_output = cnn_layer(sentence_embedding_output)

    pool_layer = keras.layers.GlobalMaxPool1D()

    relation_embedding_output = relation_embedding(relation_input)

    relation_embedding_output = Lambda(lambda x: tf.squeeze(x[0:1, :, :], axis=0))(relation_embedding_output)
    relation_embedding_output = Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(relation_embedding_output)

    # print(Lambda(lambda x : tf.squeeze(x[0:1, :, :], axis=0)).compute_output_shape((None, 19, 1000)))

    # output = Dropout(0.3)(cnn_output)
    # None, 1000
    cnn_output = pool_layer(cnn_output)

    output = cnn_output

    # output = keras.layers.Dropout(0.5)(output)
    output = Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(output)

    output = LossLayer()([output, relation_embedding_output, label_input])

    # output = Lambda(tensordot)([output, relation_embedding_output])

    # output = Activation(activation="softmax")(output)
    #    output = Dense(RELATION_COUNT, activation="softmax")(output)

    model = keras.Model(
        inputs=[index_input, relative_e1_index_input, relative_e2_index_input, relation_input, label_input],
        outputs=[output])

    sgd = SGD(0.03)
    model.compile(optimizer=sgd, loss=None, metrics=["accuracy"])

    model.summary()

    model.fit(x=[train_index, train_relative_e1_pos, train_relative_e2_pos, train_relation_input_arr, train_labels],
              y=None,
              # validation_split=0.1,
              batch_size=32,
              epochs=100,
              shuffle=True,
              callbacks=[f1_calculator(test_index, test_relative_e1_pos, test_relative_e2_pos, test_relation_input_arr,
                                       test_labels),
                         ModelCheckpoint("simple_cnn.model", "f1_score", 0, True, False, "max"),
                         EarlyStopping("f1_score", 0.000001, 20, 0, "max")
                         ]
              )
