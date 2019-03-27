# coding=utf-8
import pandas as pd
import re
import json
import numpy as np
import nltk
import copy
from collections import Counter
import os

# 实际上的最大长度为97
FIXED_SIZE = 96

TRAIN_DATA_FILE = "/Users/zhouyang/datasets/SemEval2010_task8_all_data/" \
                  "SemEval2010_task8_training/TRAIN_FILE.TXT"
TEST_DATA_FILE = "/Users/zhouyang/datasets/SemEval2010_task8_all_data/" \
                 "SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"

relations_dict = {
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

RELATION_COUNT = len(relations_dict)

pos_type = {
    "CC": 0,
    "CD": 1,
    "DT": 2,
    "EX": 3,
    "FW": 4,
    "IN": 5,
    "JJ": 6,
    "JJR": 7,
    "JJS": 8,
    "LS": 9,
    "MD": 10,
    "NN": 11,
    "NNS": 12,
    "NNP": 13,
    "NNPS": 14,
    "PDT": 15,
    "POS": 16,
    "PRP": 17,
    "PRP$": 18,
    "RB": 19,
    "RBR": 20,
    "RBS": 21,
    "RP": 22,
    "SYM": 23,
    "TO": 24,
    "UH": 25,
    "VB": 26,
    "VBD": 27,
    "VBG": 28,
    "VBN": 29,
    "VBP": 30,
    "VBZ": 31,
    "WDT": 32,
    "WP": 33,
    "WP$": 34,
    "WRB": 35,
    "NONE": 36,
    "BLANK": 37,
}

POS_EMBEDDING_LENGTH = len(pos_type)
PF_EMBEDDING_LENGTH = 2 * FIXED_SIZE


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
    string = re.sub(r"\?", " \\? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def get_entity_pos(words):
    e1_begin = -1
    e1_end = -1
    e2_begin = -1
    e2_end = -1
    length = len(words)
    for i in range(length):
        if words[i] == "e1" and e1_begin == -1:
            e1_begin = i
        elif words[i] == "e1" and e1_end == -1:
            e1_end = i
        elif words[i] == "e1":
            assert e1_begin != -1 and e1_end != -1
        if words[i] == "e2" and e2_begin == -1:
            e2_begin = i
        elif words[i] == "e2" and e2_end == -1:
            e2_end = i
        elif words[i] == "e2":
            assert e2_begin != -1 and e2_end != -1
    return e1_begin, e1_end, e2_begin, e2_end


def remove_indicator(words):
    for _ in range(2):
        words.remove("e1")
        words.remove("e2")
    return words


def replace_bracket(words):
    zip_pairs = [("\(", "("), ("\)", ")")]
    for i in range(len(words)):
        for x, y in zip_pairs:
            if words[i] == x:
                words[i] = y
    return words


def append_dot(words):
    words.append(".")
    return words


def contain_annotation(words):
    try:
        if words.index("(") != -1 and words.index(")") != -1:
            return True
    except:
        return False


def get_label(labels, label_count):
    labels = list(labels)
    vec = np.zeros((len(labels), label_count))
    vec[np.arange(len(labels)), labels] = 1
    vec = list(vec)
    return vec


def extend(list1, list2):
    list1.extend(list2)
    return list1


def limit_length(df, column="words"):
    df[column] = df[column].map(lambda x: x[:FIXED_SIZE])
    df[column] = df[column].map(
        lambda x: x if len(x) >= FIXED_SIZE else extend(x, ["BLANK" for _ in range(FIXED_SIZE - len(x))]))
    return df


def limit_pos_length(df, pos="pos_tag"):
    df[pos] = df[pos].map(lambda x: x[:FIXED_SIZE])
    df[pos] = df[pos].map(
        lambda x: x if len(x) >= FIXED_SIZE else extend(x, [pos_type["BLANK"] for _ in range(FIXED_SIZE - len(x))]))
    return df


def calculate_relative_distance(max_length, entity_pos, length, entity_substruct_word=True):
    pos = list(range(max_length))
    if entity_substruct_word:
        pos = map(lambda x: entity_pos - x, pos)
    else:
        pos = map(lambda x: x - entity_pos, pos)
    pos = map(lambda x: x + max_length - 1, pos)
    # 对于超出长度的情况的处理
    for i in range(length, max_length):
        pos[i] = max_length * 2 - 1
    return pos


# todo: 检验一下去掉注释之后的结果
if __name__ == "__main__":
    train_file = open(TRAIN_DATA_FILE)
    test_file = open(TEST_DATA_FILE)

    train_lines = train_file.readlines()
    test_lines = test_file.readlines()

    TRAIN_LENGTH = len(train_lines) // 4

    train_lines.extend(test_lines)

    all_lines = train_lines
    all_lines = map(lambda x: x.strip(), all_lines)
    sentences = all_lines[0::4]

    copy_sentences = copy.deepcopy(sentences)

    sentences = map(lambda x: x.split("\t")[1][1:-1], sentences)
    sentences = map(lambda x: clean_str(x), sentences)
    relations = all_lines[1::4]
    words = map(lambda x: x.split(" "), sentences)
    entity_pos_list = map(lambda x: get_entity_pos(x), words)
    e1_begin_pos = map(lambda x: x[0], entity_pos_list)
    e1_end_pos = map(lambda x: x[1] - 2, entity_pos_list)
    e2_begin_pos = map(lambda x: x[2] - 2, entity_pos_list)
    e2_end_pos = map(lambda x: x[3] - 4, entity_pos_list)
    words = map(lambda x: remove_indicator(x), words)
    words = map(lambda x: replace_bracket(x), words)
    words = map(lambda x: append_dot(x), words)
    sentences = map(lambda x: " ".join(x), words)
    pos_tag = map(lambda x: nltk.pos_tag(x), words)
    pos_tag = map(lambda x: map(lambda y: y[1], x), pos_tag)
    pos_tag = map(lambda x: map(lambda y: pos_type[y] if y in pos_type else pos_type["NONE"], x), pos_tag)
    relations = map(lambda x: relations_dict[x], relations)

    df = pd.DataFrame(
        columns=["sentences", "words", "e1_begin_pos", "e1_end_pos", "e2_begin_pos", "e2_end_pos", "pos_tag",
                 "relations"])

    df["sentences"] = sentences
    df["words"] = words
    df["e1_begin_pos"] = e1_begin_pos
    df["e1_end_pos"] = e1_end_pos
    df["e2_begin_pos"] = e2_begin_pos
    df["e2_end_pos"] = e2_end_pos
    df["pos_tag"] = pos_tag
    df["relations"] = relations
    df["label"] = get_label(df["relations"], RELATION_COUNT)
    df["len"] = df["words"].map(lambda x: len(x))

    word_counter = Counter()

    for words in df["words"]:
        for word in words:
            word_counter[word] += 1

    words = word_counter.most_common()

    words_dict = {w[0]: index + 1 for (index, w) in enumerate(words)}
    words_dict["BLANK"] = 0

    df = limit_length(df)
    df = limit_pos_length(df)

    df["words"] = df["words"].map(lambda x: map(lambda y: words_dict[y], x))
    df["relative_e1_distance"] = df.apply(
        lambda row: calculate_relative_distance(FIXED_SIZE, row["e1_end_pos"], row["len"]), axis=1)
    df["relative_e2_distance"] = df.apply(
        lambda row: calculate_relative_distance(FIXED_SIZE, row["e2_end_pos"], row["len"]), axis=1)

    df_train = df[:TRAIN_LENGTH]
    df_test = df[TRAIN_LENGTH:]

    train_word_index = np.array(df_train["words"].tolist())
    test_word_index = np.array(df_test["words"].tolist())

    train_relative_e1_distance = np.array(df_train["relative_e1_distance"].tolist())
    train_relative_e2_distance = np.array(df_train["relative_e2_distance"].tolist())

    test_relative_e1_distance = np.array(df_test["relative_e1_distance"].tolist())
    test_relative_e2_distance = np.array(df_test["relative_e2_distance"].tolist())

    train_pos_tag = np.array(df_train["pos_tag"].tolist())
    test_pos_tag = np.array(df_test["pos_tag"].tolist())

    train_e1_pos = np.array(df_train["e1_end_pos"].tolist())
    test_e1_pos = np.array(df_test["e1_end_pos"].tolist())

    train_e2_pos = np.array(df_train["e2_end_pos"].tolist())
    test_e2_pos = np.array(df_test["e2_end_pos"].tolist())

    train_label = np.array(df_train["label"].tolist())
    test_label = np.array(df_test["label"].tolist())

    store_files = [
        (train_word_index, "train_word_index.npy"),
        (test_word_index, "test_word_index.npy"),
        (train_relative_e1_distance, "train_relative_e1_distance.npy"),
        (test_relative_e1_distance, "test_relative_e1_distance.npy"),
        (train_relative_e2_distance, "train_relative_e2_distance.npy"),
        (test_relative_e2_distance, "test_relative_e2_distance.npy"),
        (train_pos_tag, "train_pos_tag.npy"),
        (test_pos_tag, "test_pos_tag.npy"),
        (train_e1_pos, "train_e1_pos.npy"),
        (test_e1_pos, "test_e1_pos.npy"),
        (train_e2_pos, "train_e2_pos.npy"),
        (test_e2_pos, "test_e2_pos.npy"),
        (train_label, "train_label.npy"),
        (test_label, "test_label.npy"),
    ]

    os.chdir("/home/zy/data/zy_paper/common_pretreat")

    for arr, name in store_files:
        np.save(name, arr)

    file = open("word_list.json", "w")

    json.dump(words_dict, file)
