# coding:utf-8
import numpy as np
import os
import keras
from keras.layers import Dense, Conv1D, Dropout, Input, concatenate, MaxPooling1D, Flatten, LSTM, Bidirectional, \
    RepeatVector, Reshape, TimeDistributed, Activation
from keras.layers.embeddings import Embedding
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
from keras.callbacks import EarlyStopping, ModelCheckpoint
from comm.scorer import get_marco_f1
from comm.piecewise_maxpool import piecewise_maxpool_layer
import tensorflow as tf

POS_EMBEDDING_DIM = 25
FIXED_SIZE = 101


class f1_calculator(Callback):
    def __init__(self, index, pos1_index, pos2_index, e1_pos, e2_pos, result):
        self.index = index
        self.result = result
        self.pos1_index = pos1_index
        self.pos2_index = pos2_index
        self.e1_pos = e1_pos
        self.e2_pos = e2_pos

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        predict_result = self.model.predict(x=[self.index, self.pos1_index, self.pos2_index, self.e1_pos, self.e2_pos])
        f1_score = get_marco_f1(predict_result, self.result)
        logs["f1_score"] = f1_score
        # self.save_best(f1_score)

    def save_best(self, f1):
        with open("best/README", "r") as file:
            best_f1 = file.readline()
            best_f1 = float(best_f1)
        if f1 > best_f1:
            os.system("rm best/README")
            with open("best/README", "w") as file:
                file.write(str(f1))
            self.model.save("best/simple_cnn.model")
            pass


def reshape(x):
    tensor = x[0]
    filters = x[1]
    part_num = x[2]
    batch_size = tf.shape(tensor)[0]
    return tf.reshape(tensor=tensor, shape=(batch_size, filters, part_num))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.chdir("/home/zy/data/zy_paper/task8")
    train_relative_e1_pos = np.load("train_relative_e1_pos_with_indicator.npy")
    train_relative_e2_pos = np.load("train_relative_e2_pos_with_indicator.npy")
    test_relative_e1_pos = np.load("test_relative_e1_pos_with_indicator.npy")
    test_relative_e2_pos = np.load("test_relative_e2_pos_with_indicator.npy")
    train_labels = np.load("train_labels.npy")
    test_labels = np.load("test_labels.npy")
    train_e1_pos = np.load("train_e1_pos_with_indicator.npy")
    train_e2_pos = np.load("train_e2_pos_with_indicator.npy")
    test_e1_pos = np.load("test_e1_pos_with_indicator.npy")
    test_e2_pos = np.load("test_e2_pos_with_indicator.npy")
    vec = np.load("wiki_vec.npy")
    train_index = np.load("train_wiki_with_indicator_index.npy")
    test_index = np.load("test_wiki_with_indicator_index.npy")

    index_input = Input(shape=(FIXED_SIZE,), dtype="int32")

    pos1_input = Input(shape=(FIXED_SIZE,), dtype="int32")
    pos2_input = Input(shape=(FIXED_SIZE,), dtype="int32")

    e1_pos = Input(shape=(1,), dtype="int32")
    e2_pos = Input(shape=(1,), dtype="int32")

    pos1_embedding = Embedding(input_dim=2 * FIXED_SIZE + 1, output_dim=POS_EMBEDDING_DIM, input_length=FIXED_SIZE,
                               trainable=True) \
        (pos1_input)

    pos2_embedding = Embedding(input_dim=2 * FIXED_SIZE + 1, output_dim=POS_EMBEDDING_DIM, input_length=FIXED_SIZE,
                               trainable=True) \
        (pos2_input)

    pos_embedding = concatenate([pos1_embedding, pos2_embedding], axis=2)

    word_embedding = Embedding(input_dim=len(vec), output_dim=EMBEDDING_DIM, weights=[vec], input_length=FIXED_SIZE,
                               trainable=True)(index_input)

    embedding_output = concatenate([word_embedding, pos_embedding], axis=2)
    # (None, FIXED_SIZE, 128)
    cnn1 = Conv1D(filters=128, kernel_size=2, strides=1, padding="same", activation="relu")(embedding_output)
    cnn2 = Conv1D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu")(embedding_output)
    cnn3 = Conv1D(filters=128, kernel_size=4, strides=1, padding="same", activation="relu")(embedding_output)
    cnn4 = Conv1D(filters=128, kernel_size=5, strides=1, padding="same", activation="relu")(embedding_output)
    cnn1 = piecewise_maxpool_layer(filter_num=128, fixed_size=FIXED_SIZE)([cnn1, e1_pos, e2_pos])
    cnn2 = piecewise_maxpool_layer(filter_num=128, fixed_size=FIXED_SIZE)([cnn2, e1_pos, e2_pos])
    cnn3 = piecewise_maxpool_layer(filter_num=128, fixed_size=FIXED_SIZE)([cnn3, e1_pos, e2_pos])
    cnn4 = piecewise_maxpool_layer(filter_num=128, fixed_size=FIXED_SIZE)([cnn4, e1_pos, e2_pos])

    reshape_layer = keras.layers.Lambda(lambda x : tf.reshape(tensor = x, shape = (tf.shape(x)[0], 128, 3)))

    cnn1 = reshape_layer(cnn1)
    cnn2 = reshape_layer(cnn2)
    cnn3 = reshape_layer(cnn3)
    cnn4 = reshape_layer(cnn4)


    # todo: 这里加上权重因子，前，中，后 哪个部分的权重最大
    # cnn1 = MaxPooling1D(pool_size=FIXED_SIZE, strides=1, padding="same")(cnn1)
    # cnn2 = MaxPooling1D(pool_size=FIXED_SIZE, strides=1, padding="same")(cnn2)
    # cnn3 = MaxPooling1D(pool_size=FIXED_SIZE, strides=1, padding="same")(cnn3)
    # cnn4 = MaxPooling1D(pool_size=FIXED_SIZE, strides=1, padding="same")(cnn4)
    # (None, 128 * 4, 3)
    cnn_output = concatenate([cnn1, cnn2, cnn3, cnn4], axis=1)


    # output = Flatten()(cnn_output)

    output = Dropout(rate=0.3)(cnn_output)

    # output = Dense(128, activation="sigmoid")(output)

    output = Dense(RELATION_COUNT, activation="softmax")(output)
    model = Model(inputs=[index_input, pos1_input, pos2_input, e1_pos, e2_pos], outputs=[output])

    # model = multi_gpu_model(model, gpus=4)

    optimizer = Adadelta(lr=1.0)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=[])
    model.fit(x=[train_index, train_relative_e1_pos, train_relative_e2_pos, train_e1_pos, train_e2_pos],
              y=[train_labels],
              validation_data=(
                  [test_index, test_relative_e1_pos, test_relative_e2_pos, test_e1_pos, test_e2_pos], [test_labels]),
              batch_size=32,
              epochs=100,
              callbacks=[f1_calculator(test_index, test_relative_e1_pos, test_relative_e2_pos, test_e1_pos, test_e2_pos,
                                       test_labels),
                         ModelCheckpoint("simple_cnn.model", "f1_score", 0, True, False, "max"),
                         EarlyStopping("f1_score", 0.000001, 20, 0, "max")
                         ]
              )
