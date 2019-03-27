# coding=utf-8

# coding:utf-8

import random
from numpy.random import seed
from tensorflow import set_random_seed

seed(1337)
random.seed(1337)
set_random_seed(1337)

import numpy as np
import tensorflow as tf

import os
import keras
from keras.layers import Dense, Conv1D, Dropout, Input, concatenate, MaxPooling1D, Flatten, LSTM, Bidirectional, \
    RepeatVector, Reshape, TimeDistributed, Activation, MaxPool1D, Lambda, BatchNormalization, GlobalMaxPool1D
from keras.layers.embeddings import Embedding
from keras.layers import GRU
from keras.engine.topology import Layer
from keras.layers.merge import concatenate, add, multiply, dot
from keras import Model
import sys

sys.path.append("..")
from pretreat.semeval_2010 import EMBEDDING_DIM
from pretreat.semeval_2010 import RELATION_COUNT

from keras.optimizers import Adadelta, Adam
from keras.optimizers import sgd
from keras.callbacks import Callback
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from comm.scorer import get_marco_f1
from keras.metrics import mean_squared_error, categorical_crossentropy, mse, mae
from keras.layers.advanced_activations import LeakyReLU, PReLU
from comm.piecewise_maxpool import piecewise_maxpool_layer
from keras.utils import multi_gpu_model
from comm.marco_f1 import f1
from keras.constraints import max_norm
import tensorflow as tf
from keras import backend as K
from pretreat.semeval_2010_google import POS_COUNT

POS_EMBEDDING_DIM = 150
FIXED_SIZE = 100


class f1_calculator(Callback):
    def __init__(self, index, pos1_index, pos2_index, result, test_pos):
        self.index = index
        self.result = result
        self.pos1_index = pos1_index
        self.pos2_index = pos2_index
        self.test_pos = test_pos

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        predict_result = self.model.predict(x=[self.index, self.pos1_index, self.pos2_index, self.test_pos])
        f1_score = get_marco_f1(predict_result, self.result)
        self.save_best(f1_score)
        logs["f1_score"] = f1_score

    def save_best(self, f1):
        with open("/home/zy/git/zy_paper/cnn/simple_best/README", "r") as file:
            best_f1 = file.readline()
            best_f1 = float(best_f1)
        if f1 > best_f1:
            os.system("rm /home/zy/git/zy_paper/cnn/simple_best/README")
            with open("/home/zy/git/zy_paper/cnn/simple_best/README", "w") as file:
                file.write(str(f1))
            self.model.save("/home/zy/git/zy_paper/cnn/simple_best/simple_cnn.model")


def highway(x):
    g = x[0]
    t = x[1]
    x = x[2]
    return t * g + (1. - t) * x


# sliding window有什么更好的实现方式？
class ConvInputLayer(Layer):
    """
    Distribute word vectors into chunks - input for the convolution operation
    Input dim: [batch_size x sentence_len x word_vec_dim]
    Output dim: [batch_size x (sentence_len - filter_width + 1) x filter_width x word_vec_dim]
    """

    def __init__(self, filter_width, sent_len, **kwargs):
        super(ConvInputLayer, self).__init__(**kwargs)
        self.filter_width = filter_width
        self.sent_len = sent_len

    def call(self, x):
        chunks = []
        for i in xrange(self.sent_len - self.filter_width + 1):
            chunk = x[:, i:i + self.filter_width, :]
            chunk = K.expand_dims(chunk, 1)
            chunks.append(chunk)
        return K.concatenate(chunks, 1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.sent_len - self.filter_width + 1, self.filter_width, input_shape[-1])


def rnf_conv(x, kernel_size, hidden_dim, num_sequence):
    emb_layer = ConvInputLayer(kernel_size, num_sequence)(x)
    conv_layer = TimeDistributed(GRU(hidden_dim, dropout=0.4, recurrent_dropout=0.4, ))(emb_layer)
    output = GlobalMaxPool1D()(conv_layer)
    return output


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.chdir("/home/zy/data/zy_paper/google")
    train_relative_e1_pos = np.load("train_relative_e1_pos_without_indicator.npy")
    train_relative_e2_pos = np.load("train_relative_e2_pos_without_indicator.npy")
    test_relative_e1_pos = np.load("test_relative_e1_pos_without_indicator.npy")
    test_relative_e2_pos = np.load("test_relative_e2_pos_without_indicator.npy")
    train_labels = np.load("train_labels.npy")
    test_labels = np.load("test_labels.npy")
    # train_e1_pos = np.load("train_e1_pos_without_indicator.npy")
    # train_e2_pos = np.load("train_e2_pos_without_indicator.npy")
    # test_e1_pos = np.load("test_e1_pos_without_indicator.npy")
    # test_e2_pos = np.load("test_e2_pos_without_indicator.npy")
    vec = np.load("google_vec.npy")
    train_index = np.load("train_google_without_indicator_index.npy")
    test_index = np.load("test_google_without_indicator_index.npy")

    train_sent = np.load("/home/zy/train.npy")
    test_sent = np.load("/home/zy/test.npy")

    # index_input = Input(shape=(FIXED_SIZE,), dtype="int32")

    sentence_input = Input(shape=(FIXED_SIZE, 1024), dtype="float32")

    pos1_input = Input(shape=(FIXED_SIZE,), dtype="int32")
    pos2_input = Input(shape=(FIXED_SIZE,), dtype="int32")

    pos_tag_input = Input(shape=(FIXED_SIZE,), dtype="int32")

    train_pos = np.load("train_pos_tag_without_indicator.npy")
    test_pos = np.load("test_pos_tag_without_indicator.npy")

    # e1_pos = Input(shape=(1,), dtype="int32")
    # e2_pos = Input(shape=(1,), dtype="int32")

    pos1_embedding = Embedding(input_dim=2 * FIXED_SIZE - 1, output_dim=POS_EMBEDDING_DIM, input_length=FIXED_SIZE,
                               trainable=True) \
        (pos1_input)

    pos2_embedding = Embedding(input_dim=2 * FIXED_SIZE - 1, output_dim=POS_EMBEDDING_DIM, input_length=FIXED_SIZE,
                               trainable=True) \
        (pos2_input)

    pos_tag_embedding = Embedding(input_dim=POS_COUNT, output_dim=50, input_length=FIXED_SIZE,
                                  trainable=True)(pos_tag_input)

    pos_embedding = concatenate([pos1_embedding, pos2_embedding], axis=2)

    word_embedding = sentence_input
    # word_embedding = Embedding(input_dim=len(vec), output_dim=EMBEDDING_DIM, weights=[vec], input_length=FIXED_SIZE,
    #                            trainable=False)(index_input)

    embedding_output = concatenate([word_embedding, pos_embedding, pos_tag_embedding], axis=2)
    # embedding_output = add([word_embedding, pos_embedding])

    # cnn1 = rnf_conv(embedding_output, 3, 150, FIXED_SIZE)
    cnn2 = rnf_conv(embedding_output, 3, 200, FIXED_SIZE)
    cnn3 = rnf_conv(embedding_output, 4, 200, FIXED_SIZE)
    cnn4 = rnf_conv(embedding_output, 5, 200, FIXED_SIZE)
    # cnn4 = rnf_conv(embedding_output, 6, 200, FIXED_SIZE)

    # cnn1 = Conv1D(filters=150, kernel_size=2, strides=1, padding="same", activation="relu")(embedding_output)
    # cnn2 = Conv1D(filters=150, kernel_size=3, strides=1, padding="same", activation="relu")(embedding_output)
    # cnn3 = Conv1D(filters=150, kernel_size=4, strides=1, padding="same", activation="relu")(embedding_output)
    # cnn4 = Conv1D(filters=150, kernel_size=5, strides=1, padding="same", activation="relu")(embedding_output)

    cnn_output = concatenate([ cnn2, cnn3, cnn4], axis=-1)

    # cnn_output = MaxPooling1D(pool_size=FIXED_SIZE, strides=FIXED_SIZE, padding="valid")(cnn_output)
    # cnn_output = GlobalMaxPool1D()(cnn_output)

    # g = Dense(600, activation="relu")(cnn_output) highway network没有用
    # t = Dense(600, activation=None)(cnn_output)
    # t = Lambda(lambda x: tf.nn.sigmoid(x + 2.0))(t)
    #
    # cnn_output = Lambda(highway)([g, t, cnn_output])

    # cnn_output = BatchNormalization()(cnn_output)

    # cnn_output = Lambda(lambda x: tf.squeeze(x, axis=1))(cnn_output)

    # cnn1 = MaxPooling1D(pool_size=FIXED_SIZE, strides=1, padding="same")(cnn1)
    # cnn2 = MaxPooling1D(pool_size=FIXED_SIZE, strides=1, padding="same")(cnn2)
    # cnn3 = MaxPooling1D(pool_size=FIXED_SIZE, strides=1, padding="same")(cnn3)
    # cnn4 = MaxPooling1D(pool_size=FIXED_SIZE, strides=1, padding="same")(cnn4)

    # cnn_output = concatenate([cnn1, cnn2, cnn3, cnn4], axis=1)

    # output = Flatten()(cnn_output)

    output = Dropout(rate=0.5)(cnn_output)

    # output = Dense(128, activation="tanh")(output)

    output = Dense(RELATION_COUNT, activation="softmax", kernel_regularizer=keras.regularizers.l2(0.001),
                   bias_regularizer=keras.regularizers.l2(0.001))(output)

    model = Model(inputs=[sentence_input, pos1_input, pos2_input, pos_tag_input], outputs=[output])

    model.summary()

    # model = multi_gpu_model(model, gpus=4)

    optimizer = Adam()

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(x=[train_sent, train_relative_e1_pos, train_relative_e2_pos, train_pos],
              y=[train_labels],
              validation_data=(
                  [test_sent, test_relative_e1_pos, test_relative_e2_pos, test_pos], [test_labels]),
              batch_size=42,
              epochs=100,
              callbacks=[f1_calculator(test_sent, test_relative_e1_pos, test_relative_e2_pos,
                                       test_labels, test_pos),
                         ModelCheckpoint("simple_cnn.model", "f1_score", 0, True, False, "max"),
                         EarlyStopping("f1_score", 0.000001, 20, 0, "max")
                         ]
              )

# 加上pos tag，没有显著提升效果
# 这个创新点比较牵强
