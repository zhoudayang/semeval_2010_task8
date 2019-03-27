# coding=utf-8
import keras.backend as K
import tensorflow as tf
from keras.layers import Embedding, Dense, Conv1D, Input, concatenate, Conv2D, Lambda, Dropout, Concatenate
import numpy as np
import os
from pretreat.semeval_2010_google import FIXED_SIZE, RELATION_COUNT, EMBEDDING_DIM
from comm.piecewise_maxpool import piecewise_maxpool_layer
from keras.optimizers import Adam, Adadelta
from comm.scorer import get_marco_f1
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

POS_EMBEDDING_DIM = 50
from keras import Model


# todo: 将reshape更换为transpose
def reshape(tensor):
    return tf.expand_dims(input=tensor, axis=1)
    # shape = tf.shape(tensor)
    # batch_size = shape[0]
    # return tf.reshape(tensor=tensor, shape=(batch_size, 1, FIXED_SIZE, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM))


def reshape2(tensor):
    return tf.transpose(tensor, perm=[0, 2, 1])
    # shape = tf.shape(tensor)
    # batch_size = shape[0]
    # size1 = shape[1]
    # size2 = shape[2]
    # return tf.reshape(tensor=tensor, shape=(batch_size, size2, size1))


class f1_calculator(Callback):
    def __init__(self, google_index, fasttext_index, relative_pos1, relative_pos2, e1_pos, e2_pos, result):
        self.google_index = google_index
        self.fasttext_index = fasttext_index
        self.relative_pos1 = relative_pos1
        self.relative_pos2 = relative_pos2
        self.e1_pos = e1_pos
        self.e2_pos = e2_pos
        self.result = result

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        predict_result = self.model.predict(
            x=[self.google_index, self.fasttext_index, self.relative_pos1, self.relative_pos2, self.e1_pos,
               self.e2_pos])
        f1_score = get_marco_f1(predict_result, self.result)
        logs["f1_score"] = f1_score
        self.save_best(f1_score)

    def save_best(self, f1):
        with open("/home/zy/git/zy_paper/cnn_multi_channel/best/README", "r") as file:
            best_f1 = file.readline()
            best_f1 = float(best_f1)
        if f1 > best_f1:
            os.system("rm /home/zy/git/zy_paper/cnn_multi_channel/best/README")
            with open("/home/zy/git/zy_paper/cnn_multi_channel/best/README", "w") as file:
                file.write(str(f1))
            self.model.save("/home/zy/git/zy_paper/cnn_multi_channel/best/cnn_multi_channel.model")


if __name__ == "__main__":
    os.chdir("/home/zy/data/zy_paper/google")
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    google_vec = np.load("google_vec.npy")
    train_labels = np.load("train_labels.npy")
    test_labels = np.load("test_labels.npy")
    train_google_index = np.load("train_google_without_indicator_index.npy")
    test_google_index = np.load("test_google_without_indicator_index.npy")
    train_relative_e1_pos = np.load("train_relative_e1_pos_without_indicator.npy")
    train_relative_e2_pos = np.load("train_relative_e2_pos_without_indicator.npy")
    test_relative_e1_pos = np.load("test_relative_e1_pos_without_indicator.npy")
    test_relative_e2_pos = np.load("test_relative_e2_pos_without_indicator.npy")
    os.chdir("/home/zy/data/zy_paper/glove")
    train_e1_pos = np.load("train_e1_pos_without_indicator.npy")
    train_e2_pos = np.load("train_e2_pos_without_indicator.npy")
    test_e1_pos = np.load("test_e1_pos_without_indicator.npy")
    test_e2_pos = np.load("test_e2_pos_without_indicator.npy")
    train_fasttext_index = np.load("train_glove_without_indicator_index.npy")
    test_fasttext_index = np.load("test_glove_without_indicator_index.npy")
    fasttext_vec = np.load("glove_vec.npy")

    google_index_input = Input(shape=(FIXED_SIZE,), dtype="int32")
    fasttext_input = Input(shape=(FIXED_SIZE,), dtype="int32")

    pos1_input = Input(shape=(FIXED_SIZE,), dtype="int32")
    pos2_input = Input(shape=(FIXED_SIZE,), dtype="int32")

    e1_pos = Input(shape=(1,), dtype="int32")
    e2_pos = Input(shape=(1,), dtype="int32")

    google_embedding = Embedding(input_dim=len(google_vec), output_dim=EMBEDDING_DIM, weights=[google_vec],
                                 input_length=FIXED_SIZE, trainable=True)(google_index_input)
    fasttext_embedding = Embedding(input_dim=len(fasttext_vec), output_dim=EMBEDDING_DIM, weights=[fasttext_vec],
                                   input_length=FIXED_SIZE, trainable=True)(fasttext_input)

    pos1_embedding = Embedding(input_dim=2 * FIXED_SIZE + 1, output_dim=POS_EMBEDDING_DIM, input_length=FIXED_SIZE,
                               trainable=True)(pos1_input)
    pos2_embedding = Embedding(input_dim=2 * FIXED_SIZE + 1, output_dim=POS_EMBEDDING_DIM, input_length=FIXED_SIZE,
                               trainable=True)(pos2_input)

    google_embedding = concatenate([google_embedding, pos1_embedding, pos2_embedding], axis=2)

    fasttext_embedding = concatenate([fasttext_embedding, pos1_embedding, pos2_embedding], axis=2)

    conv_layer3_1 = Conv2D(filters=42, kernel_size=(3, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM),
                           strides=(1, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM), padding="same",
                           data_format="channels_first", activation="relu")

    conv_layer3_2 = Conv2D(filters=42, kernel_size=(3, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM),
                           strides=(1, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM), padding="same",
                           data_format="channels_first", activation="relu")

    conv_layer3_3 = Conv2D(filters=42, kernel_size=(4, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM),
                           strides=(1, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM), padding="same",
                           data_format="channels_first", activation="relu")

    conv_layer4_1 = Conv2D(filters=42, kernel_size=(4, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM),
                           strides=(1, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM), padding="same",
                           data_format="channels_first", activation="relu")

    conv_layer4_2 = Conv2D(filters=42, kernel_size=(4, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM),
                           strides=(1, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM), padding="same",
                           data_format="channels_first", activation="relu")

    conv_layer4_3 = Conv2D(filters=42, kernel_size=(4, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM),
                           strides=(1, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM), padding="same",
                           data_format="channels_first", activation="relu")

    conv_layer5_1 = Conv2D(filters=42, kernel_size=(5, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM),
                           strides=(1, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM), padding="same",
                           data_format="channels_first", activation="relu")

    conv_layer5_2 = Conv2D(filters=42, kernel_size=(5, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM),
                           strides=(1, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM), padding="same",
                           data_format="channels_first", activation="relu")

    conv_layer5_3 = Conv2D(filters=42, kernel_size=(5, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM),
                           strides=(1, EMBEDDING_DIM + 2 * POS_EMBEDDING_DIM), padding="same",
                           data_format="channels_first", activation="relu")

    google_embedding = Lambda(reshape)(google_embedding)

    fasttext_embedding = Lambda(reshape)(fasttext_embedding)

    merge_embedding = concatenate([google_embedding, fasttext_embedding], axis=1)

    squeeze_layer = Lambda(lambda x: tf.squeeze(input=x, axis=-1))

    # print conv_layer3_1.compute_output_shape((None, 1, FIXED_SIZE, 400))
    # print squeeze_layer.compute_output_shape((None, 40, 101, 1))
    conv3_1_output = conv_layer3_1(google_embedding)
    conv3_1_output = squeeze_layer(conv3_1_output)
    conv3_2_output = conv_layer3_2(merge_embedding)
    conv3_2_output = squeeze_layer(conv3_2_output)
    conv3_3_output = conv_layer3_3(fasttext_embedding)
    conv3_3_output = squeeze_layer(conv3_3_output)

    conv4_1_output = conv_layer4_1(google_embedding)
    conv4_1_output = squeeze_layer(conv4_1_output)
    conv4_2_output = conv_layer4_2(merge_embedding)
    conv4_2_output = squeeze_layer(conv4_2_output)
    conv4_3_output = conv_layer4_3(fasttext_embedding)
    conv4_3_output = squeeze_layer(conv4_3_output)

    conv5_1_output = conv_layer5_1(google_embedding)
    conv5_1_output = squeeze_layer(conv5_1_output)
    conv5_2_output = conv_layer5_2(merge_embedding)
    conv5_2_output = squeeze_layer(conv5_2_output)
    conv5_3_output = conv_layer5_3(fasttext_embedding)
    conv5_3_output = squeeze_layer(conv5_3_output)

    # print Concatenate(axis=1).compute_output_shape([(None, 40, 101) for i in range(9)])
    cnn_output = concatenate(
        [conv3_1_output, conv3_2_output, conv3_3_output, conv4_1_output, conv4_2_output,
         conv4_3_output, conv5_1_output, conv5_2_output, conv5_3_output], axis=1)
    cnn_output = Lambda(reshape2)(cnn_output)
    cnn_output = piecewise_maxpool_layer(filter_num=42 * 3 * 3, fixed_size=FIXED_SIZE)([cnn_output, e1_pos, e2_pos])

    cnn_output = Dropout(rate=0.5)(cnn_output)

    output = cnn_output

    output = Dense(256, activation="sigmoid")(output)

    output = Dense(RELATION_COUNT, activation="softmax")(output)

    model = Model(inputs=[google_index_input, fasttext_input, pos1_input, pos2_input, e1_pos, e2_pos], outputs=[output])

    ada = Adadelta(lr=0.1, epsilon=1e-4)
    model.compile(optimizer=ada, loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(x=[train_google_index, train_fasttext_index, train_relative_e1_pos, train_relative_e2_pos, train_e1_pos,
                 train_e2_pos], y=[train_labels],
              # validation_split=0.1,
              batch_size=32,
              epochs=100,
              callbacks=[
                  f1_calculator(test_google_index, test_fasttext_index, test_relative_e1_pos, test_relative_e2_pos,
                                test_e1_pos, test_e2_pos, test_labels),
                  ModelCheckpoint("simple_cnn.model", "f1_score", 0, True, False, "max"),
                  EarlyStopping("f1_score", 0.000001, 20, 0, "max")
              ])
