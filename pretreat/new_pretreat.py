# coding:utf-8
import pandas as pd
import numpy as np
import json
import re
import nltk
import os
from wikipedia2vec import Wikipedia2Vec

### !!! 采用的是python3版本
"""
max : 98
25% : 12
50% : 17
75% : 23
99% : 45
99.5% : 52
99.7% : 57
99.8% : 61
99.9% : 68
99.95% : 71

e2_pos_end max : 70
"""

# todo: 对于entity范围内的词，其relative pos设为0

FIXED_SIZE = 100
EMBEDDING_DIM = 300
USE_E_SUBSTRUCT_WORD_POS = True
# 是否对于在entity范围内的词，进行特殊处理
LIMIT_ENTITY_POS = False
TRAIN_LEN = 8000
# 位置向量的总长度1000 -> 999
POSITION_EMBEDDING_LENGTH = 1000

Relations = {
    'Cause-Effect(e1,e2)': 0, 'Cause-Effect(e2,e1)': 1,
    'Component-Whole(e1,e2)': 2, 'Component-Whole(e2,e1)': 3,
    'Content-Container(e1,e2)': 4, 'Content-Container(e2,e1)': 5,
    'Entity-Destination(e1,e2)': 6, 'Entity-Destination(e2,e1)': 7,
    'Entity-Origin(e1,e2)': 8, 'Entity-Origin(e2,e1)': 9,
    'Instrument-Agency(e1,e2)': 10, 'Instrument-Agency(e2,e1)': 11,
    'Member-Collection(e1,e2)': 12, 'Member-Collection(e2,e1)': 13,
    'Message-Topic(e1,e2)': 14, 'Message-Topic(e2,e1)': 15,
    'Product-Producer(e1,e2)': 16, 'Product-Producer(e2,e1)': 17,
    'Other': 18
}


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def get_label(labels, label_count):
    labels = list(labels)
    vec = np.zeros((len(labels), label_count))
    vec[np.arange(len(labels)), labels] = 1
    vec = list(vec)
    return vec


def get_data_distributation(data, percent):
    data = list(data)
    data.sort()
    length = len(data)
    index = int(length * percent / 100.0)
    return data[index]


def extend(input_list, extend_list):
    input_list.extend(extend_list)
    return input_list


def get_relative_pos(e_pos, length, max_length, e1_begin, e1_end, e2_begin, e2_end):
    result = []
    for i in range(length):
        index = i
        if LIMIT_ENTITY_POS:
            if index >= e1_begin and index <= e1_end:
                index = e1_end
            elif index >= e2_begin and index <= e2_end:
                index = e2_end
        if USE_E_SUBSTRUCT_WORD_POS:
            result.append(max_length - 1 + e_pos - index)
        else:
            result.append(max_length - 1 + index - e_pos)
    result = result[:max_length]
    if length < max_length:
        for _ in range(max_length - length):
            result.append(999)
    return result


def get_random_vec():
    vec = np.random.rand(EMBEDDING_DIM)
    norm = np.sum(vec ** 2) ** 0.5
    normalized = vec / norm
    return normalized


def get_word_index(word_dict, word_list):
    result = list(map(lambda x: word_dict[x], word_list))
    return result


if __name__ == "__main__":
    all_file = open("../data/all.txt", "r")
    file_lines = all_file.readlines()
    lines = file_lines[0::4]
    relations = file_lines[1::4]

    lmap = lambda func, *iterable: list(map(func, *iterable))

    lines = lmap(lambda x: x.strip(), lines)
    lines = lmap(lambda x: x.split("\t")[1], lines)
    lines = lmap(lambda x: x[1:-1], lines)
    relations = lmap(lambda x: x.strip(), relations)
    relations = lmap(lambda x: Relations[x], relations)

    map_pairs = [("<e1>", " _e1_ "), ("</e1>", " _/e1_ "), ("<e2>", " _e2_ "), ("</e2>", " _/e2_ ")]

    for pair in map_pairs:
        lines = lmap(lambda x: x.replace(pair[0], pair[1]), lines)

    tokens = lmap(lambda x: nltk.word_tokenize(x), lines)

    e1_pos_begin = lmap(lambda x: x.index("_e1_"), tokens)
    e1_pos_end = lmap(lambda x: x.index("_/e1_") - 2, tokens)

    e2_pos_begin = lmap(lambda x: x.index("_e2_") - 2, tokens)
    e2_pos_end = lmap(lambda x: x.index("_/e2_") - 4, tokens)

    for token in ["_e1_", "_/e1_", "_e2_", "_/e2_"]:
        for row in tokens:
            row.remove(token)
    lines = lmap(lambda x: " ".join(x), tokens)
    lines = lmap(lambda x: clean_str(x), lines)

    df = pd.DataFrame(np.array([lines, e1_pos_begin, e1_pos_end, e2_pos_begin, e2_pos_end, relations]).T,
                      columns=["sentence", "e1_pos_begin", "e1_pos_end", "e2_pos_begin", "e2_pos_end", "relation"])
    df["words"] = df["sentence"].map(lambda x: nltk.word_tokenize(x))
    for column in ["relation", "e1_pos_begin", "e1_pos_end", "e2_pos_begin", "e2_pos_end"]:
        df[column] = pd.to_numeric(df[column])
    df["label"] = get_label(df["relation"].tolist(), len(Relations))
    df["len"] = df["words"].map(lambda x: len(x))
    df["words"] = df["words"].map(lambda x: extend(x, ["."]))
    df["words"] = df["words"].map(lambda x: x[:FIXED_SIZE])
    df["words"] = df["words"].map(lambda x: extend(x, ["BLANK" for _ in range(FIXED_SIZE - len(x))]))
    df["e1"] = df["e1_pos_end"]
    df["e2"] = df["e2_pos_end"]

    # 决定使用wiki百科语料训练的词向量来进行句子表示
    os.chdir("/home/zy/data/wiki_win10_300d_20180420")
    wiki_model = Wikipedia2Vec.load("enwiki_20180420_win10_300d.pkl")

    all_words = set()

    for i in range(len(df)):
        words = set(df["words"][i])
        all_words = all_words.union(words)

    word_to_index = {}
    vec_list = []
    index = 0
    unrecord_word_cnt = 0
    for word in all_words:
        if word == "BLANK":
            vec_list.append(np.zeros(shape=(EMBEDDING_DIM,), dtype="float32"))
            word_to_index[word] = index
        else:
            try:
                vec_list.append(wiki_model.get_word_vector(word))
                word_to_index[word] = index
            except KeyError as e:
                vec = get_random_vec()
                vec_list.append(vec)
                word_to_index[word] = index
                unrecord_word_cnt += 1
        index += 1

    word_vec = np.array(vec_list)

    df["index"] = df["words"].map(lambda x: get_word_index(word_to_index, x))
    # 此处使用wiki语料训练的词向量，未登录词汇数量为494
    # print (unrecord_word_cnt)
    # 使用统一的index表示越界词汇
    df["relative_e1_pos"] = df.apply(
        lambda row: get_relative_pos(row["e1"], row["len"], FIXED_SIZE, row["e1_pos_begin"], row["e1_pos_end"],
                                     row["e2_pos_begin"], row["e2_pos_end"]), axis=1)
    df["relative_e2_pos"] = df.apply(
        lambda row: get_relative_pos(row["e2"], row["len"], FIXED_SIZE, row["e1_pos_begin"], row["e1_pos_end"],
                                     row["e2_pos_begin"], row["e2_pos_end"]), axis=1)

    os.chdir("/home/zy/data/zy_paper/new")

    df_train = df[:TRAIN_LEN]
    df_test = df[TRAIN_LEN:]

    train_word_index = np.array(df_train["index"].tolist())
    test_word_index = np.array(df_test["index"].tolist())

    train_e1_pos = np.array(df_train["e1"].tolist())
    test_e1_pos = np.array(df_test["e1"].tolist())

    train_e2_pos = np.array(df_train["e2"].tolist())
    test_e2_pos = np.array(df_test["e2"].tolist())

    train_relative_e1_pos = np.array(df_train["relative_e1_pos"].tolist())
    test_relative_e1_pos = np.array(df_test["relative_e1_pos"].tolist())
    train_relative_e2_pos = np.array(df_train["relative_e2_pos"].tolist())
    test_relative_e2_pos = np.array(df_test["relative_e2_pos"].tolist())

    train_labels = np.array(df_train["label"].tolist())
    test_labels = np.array(df_test["label"].tolist())

    np.save("word_vec.npy", word_vec)
    np.save("train_word_index.npy", train_word_index)
    np.save("test_word_index.npy", test_word_index)
    np.save("train_e1_pos.npy", train_e1_pos)
    np.save("test_e1_pos.npy", test_e1_pos)
    np.save("train_e2_pos.npy", train_e2_pos)
    np.save("test_e2_pos.npy", test_e2_pos)
    np.save("train_relative_e1_pos.npy", train_relative_e1_pos)
    np.save("test_relative_e1_pos.npy", test_relative_e1_pos)
    np.save("train_relative_e2_pos.npy", train_relative_e2_pos)
    np.save("test_relative_e2_pos.npy", test_relative_e2_pos)
    np.save("train_labels.npy", train_labels)
    np.save("test_labels.npy", test_labels)
