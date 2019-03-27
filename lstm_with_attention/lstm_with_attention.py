# coding=utf-8
import sys

sys.path.append("..")
import numpy as np
import keras
import os
from keras.layers import Input, Embedding, LSTM, Dense
from pretreat.semeval_2010 import FIXED_SIZE, EMBEDDING_DIM, RELATION_COUNT
import keras.backend as K
import tensorflow as tf
from keras.optimizers import Adadelta
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint
from comm.scorer import get_marco_f1
from keras import regularizers
from keras.layers.merge import concatenate
from keras.utils import multi_gpu_model

POS_EMBEDDING_DIM = 100


def dot(tensor_list):
    x = tensor_list[0]
    y = tensor_list[1]
    return tf.matmul(x, y)


class f1_calculator(Callback):
    def __init__(self, index, e1_pos, e2_pos, result):
        self.index = index
        self.result = result
        self.e1_pos = e1_pos
        self.e2_pos = e2_pos

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        predict_result = self.model.predict(x=[self.index, self.e1_pos, self.e2_pos])
        f1_score = get_marco_f1(predict_result, self.result)
        logs["f1_score"] = f1_score
        self.save_best(f1_score)

    def save_best(self, f1):
        with open("/home/zy/git/zy_paper/lstm_with_attention/best/README", "r") as file:
            best_f1 = file.readline()
            best_f1 = float(best_f1)
        if f1 > best_f1:
            os.system("rm /home/zy/git/zy_paper/lstm_with_attention/best/README")
            with open("/home/zy/git/zy_paper/lstm_with_attention/best/README", "w") as file:
                file.write(str(f1))
            self.model.save("/home/zy/git/zy_paper/lstm_with_attention/best/simple_cnn.model")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.chdir("/home/zy/data/zy_paper/task8")
    train_labels = np.load("train_labels.npy")
    test_labels = np.load("test_labels.npy")
    train_index = np.load("train_wiki_with_indicator_index.npy")
    test_index = np.load("test_wiki_with_indicator_index.npy")
    train_e1_relative_pos = np.load("train_relative_e1_pos_with_indicator.npy")
    train_e2_relative_pos = np.load("train_relative_e2_pos_with_indicator.npy")
    test_e1_relative_pos = np.load("test_relative_e1_pos_with_indicator.npy")
    test_e2_relative_pos = np.load("test_relative_e2_pos_with_indicator.npy")
    vec = np.load("wiki_vec.npy")

    index_input = Input(shape=(FIXED_SIZE,), dtype="int32")
    pos1_input = Input(shape=(FIXED_SIZE,), dtype="int32")
    pos2_input = Input(shape=(FIXED_SIZE,), dtype="int32")
    pos_embedding_layer = Embedding(input_dim=2 * FIXED_SIZE + 1, output_dim=POS_EMBEDDING_DIM, input_length=FIXED_SIZE,
                                    trainable=True)

    embedding_layer = Embedding(input_dim=len(vec), output_dim=EMBEDDING_DIM, weights=[vec],
                                input_length=FIXED_SIZE, trainable=True)
    word_embedding_output = embedding_layer(index_input)
    pos1_embedding_output = pos_embedding_layer(pos1_input)
    pos2_embedding_output = pos_embedding_layer(pos2_input)

    embedding_output = concatenate([word_embedding_output, pos1_embedding_output, pos2_embedding_output], axis=2)

    forward_lstm_layer = LSTM(EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM, recurrent_activation="tanh", return_sequences=True,
                              dropout=0.3)
    backward_lstm_layer = LSTM(EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM, recurrent_activation="tanh",
                               return_sequences=True, go_backwards=True, dropout=0.3)

    forward_lstm_output = forward_lstm_layer(embedding_output)
    backward_lstm_output = backward_lstm_layer(embedding_output)

    added = keras.layers.Add()([forward_lstm_output, backward_lstm_output])

    alpha_layer = Dense(1, activation="softmax")

    alpha = alpha_layer(added)  # (None, FIXED_SIZE, 1)

    transpose_layer = keras.layers.Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1]))

    H = transpose_layer(added)  # (None, EMBEDDING_DIM, FIXED_SIZE)

    r_layer = keras.layers.Lambda(dot)
    r = r_layer([H, alpha])

    h_asterisk = keras.layers.Activation(activation="tanh")(r)

    h_asterisk = keras.layers.Lambda(lambda x: tf.squeeze(x, axis=[2]))(h_asterisk)

    output = h_asterisk

    output = Dense(256, activation="relu")(h_asterisk)

    output = keras.layers.Dropout(rate=0.5)(output)

    output = Dense(RELATION_COUNT, activation="softmax", kernel_regularizer=regularizers.l2(1e-5))(output)

    model = keras.Model(inputs=[index_input, pos1_input, pos2_input], outputs=[output])
    # model = multi_gpu_model(model, gpus=4)

    ada = Adadelta(lr=1.0)
    model.compile(optimizer=ada, loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(x=[train_index, train_e1_relative_pos, train_e2_relative_pos], y=[train_labels],
              validation_split=0.1,
              batch_size=10,
              epochs=100,
              shuffle=True,
              callbacks=[f1_calculator(test_index, test_e1_relative_pos, test_e2_relative_pos, test_labels),
                         ModelCheckpoint("simple_cnn.model", "f1_score", 0, True, False, "max"),
                         EarlyStopping("f1_score", 0.000001, 20, 0, "max")
                         ]
              )
