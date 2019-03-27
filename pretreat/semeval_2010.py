# coding=utf-8
import pandas as pd
import numpy as np
import re
import nltk
import json
import os

FIXED_SIZE = 101
EMBEDDING_DIM = 300

relations = {
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

RELATION_COUNT = len(relations)


def get_e1(sentence):
    e1_list = re.findall(r"<e1>(.+?)</e1>", sentence)
    assert (len(e1_list) == 1)
    return e1_list[0]


def get_e2(sentence):
    e2_list = re.findall(r"<e2>(.+?)</e2>", sentence)
    assert (len(e2_list) == 1)
    return e2_list[0]


def concat_indicator(words):
    ret_words = []
    i = 0
    while i < len(words):
        if words[i] == '<' and i + 2 < len(words):
            if words[i + 1] == "e1" or words[i + 1] == "e2" or words[i + 1] == "/e1" or words[i + 1] == "/e2":
                if words[i + 2] == ">":
                    ret_words.append("<" + words[i + 1] + ">")
            i += 3
        else:
            ret_words.append(words[i])
            i += 1
    return ret_words


def concat_entity(words, e1, e2):
    ret_words = []
    i = 0
    while i < len(words):
        if words[i] == "<e1>":
            ret_words.append(words[i])
            j = i + 1
            while j < len(words) and words[j] != "</e1>":
                j += 1
            ret_words.append(e1)
            i = j
            continue
        elif words[i] == "<e2>":
            ret_words.append(words[i])
            j = i + 1
            while j < len(words) and words[j] != "</e2>":
                j += 1
            ret_words.append(e2)
            i = j
            continue
        else:
            ret_words.append(words[i])
            i += 1
    return ret_words


def remove_indicator(words):
    ret_words = []
    for word in words:
        if word != "<e1>" and word != "</e1>" and word != "<e2>" and word != "</e2>":
            ret_words.append(word)
    return ret_words


def get_e1_pos(words):
    for i in range(len(words)):
        if words[i] == "<e1>":
            return i
    return -1


def get_e2_pos(words):
    for i in range(len(words)):
        if words[i] == "<e2>":
            return i
    return -1


def read_vec(json_path, vec_path):
    json_f = open(json_path, "r")
    words = json.load(json_f)
    vec = np.load(vec_path)
    word_pairs = list(words.iteritems())
    word_pairs = map(lambda x: (x[0].lower(), x[1]), word_pairs)
    lower_words = dict(word_pairs)
    vec = list(vec)
    words["UNK"] = len(vec)
    lower_words["UNK"] = len(vec)
    vec.append(np.random.normal(size=EMBEDDING_DIM, loc=0, scale=0.05))
    words["BLANK"] = len(vec)
    lower_words["BLANK"] = len(vec)
    vec.append(np.random.normal(size=EMBEDDING_DIM, loc=0, scale=0.05))
    vec = np.array(vec)
    return words, lower_words, vec


def get_word_index(word, words, lower_words):
    if word in words:
        return words[word]
    word = word.lower()
    if word in lower_words:
        return lower_words[word]
    return words["UNK"]


def extend(list1, list2):
    list1.extend(list2)
    return list1


def limit_length(df, word):
    df[word] = df[word].map(lambda x: x[:FIXED_SIZE])
    df[word] = df[word].map \
        (lambda x: x if len(x) >= FIXED_SIZE else extend(x, ["BLANK" for i in range(FIXED_SIZE - len(x))]))
    return df


def get_label(labels, label_count):
    labels = list(labels)
    vec = np.zeros((len(labels), label_count))
    vec[np.arange(len(labels)), labels] = 1
    vec = list(vec)
    return vec


if __name__ == "__main__":
    TEST_FILE = "/Users/zhouyang/datasets/SemEval2010_task8_all_data/" \
                "SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"
    TRAIN_FILE = "/Users/zhouyang/datasets/SemEval2010_task8_all_data/" \
                 "SemEval2010_task8_training/TRAIN_FILE.TXT"

    test_file = open(TEST_FILE, "r")
    train_file = open(TRAIN_FILE, "r")

    test_lines = test_file.readlines()
    train_lines = train_file.readlines()

    test_lines = map(lambda x: x.strip(), test_lines)
    train_lines = map(lambda x: x.strip(), train_lines)

    test_sentences = test_lines[::4]
    test_sentences = map(lambda x: x.split("\t")[1], test_sentences)
    test_sentences = map(lambda x: x[1:-1], test_sentences)

    train_sentences = train_lines[::4]
    train_sentences = map(lambda x: x.split("\t")[1], train_sentences)
    train_sentences = map(lambda x: x[1:-1], train_sentences)

    test_relations = test_lines[1::4]
    train_relations = train_lines[1::4]

    TRAIN_LEN = len(train_sentences)

    train_sentences.extend(test_sentences)
    train_relations.extend(test_relations)
    df = pd.DataFrame(np.array([train_sentences, train_relations]).T, columns=["sentence", "relation"])
    df["e1"] = df["sentence"].map(lambda x: get_e1(x))
    df["e2"] = df["sentence"].map(lambda x: get_e2(x))
    df["nltk_words"] = df["sentence"].map(lambda x: nltk.word_tokenize(x))
    df["nltk_words"] = df["nltk_words"].map(lambda x: concat_indicator(x))

    df["words_with_indicator"] = df.apply(lambda row: concat_entity(row["nltk_words"], row["e1"], row["e2"]), axis=1)
    df["words_without_indicator"] = df["words_with_indicator"].map(lambda x: remove_indicator(x))

    df["e1_pos"] = df["words_with_indicator"].map(lambda x: get_e1_pos(x))
    df["e2_pos"] = df["words_with_indicator"].map(lambda x: get_e2_pos(x))

    assert (len(df[df["e1_pos"] >= df["e2_pos"]]) == 0)
    df["e1_pos_with_indicator"] = df["e1_pos"] + 1
    df["e2_pos_with_indicator"] = df["e2_pos"] + 1

    df["e1_pos_without_indicator"] = df["e1_pos"]
    df["e2_pos_without_indicator"] = df["e2_pos"] - 2

    del df["e1_pos"]
    del df["e2_pos"]

    for i in range(len(df)):
        assert (df["words_with_indicator"][i][df["e1_pos_with_indicator"][i]] == df["e1"][i])
        assert (df["words_without_indicator"][i][df["e1_pos_without_indicator"][i]] == df["e1"][i])
        assert (df["words_with_indicator"][i][df["e2_pos_with_indicator"][i]] == df["e2"][i])
        assert (df["words_without_indicator"][i][df["e2_pos_without_indicator"][i]] == df["e2"][i])

    df["len_with_indicator"] = df["words_with_indicator"].map(lambda x: len(x))
    df["len_without_indicator"] = df["words_without_indicator"].map(lambda x: len(x))

    df["class"] = df["relation"].map(lambda x: relations[x])

    df["label"] = get_label(df["class"], RELATION_COUNT)

    df = limit_length(df, "words_with_indicator")
    df = limit_length(df, "words_without_indicator")

    df["relative_e1_pos_with_indicator"] = df["e1_pos_with_indicator"].map(
        lambda x: map(lambda y: FIXED_SIZE - 1 + x - y, range(FIXED_SIZE)))
    df["relative_e2_pos_with_indicator"] = df["e2_pos_with_indicator"].map(
        lambda x: map(lambda y: FIXED_SIZE - 1 + x - y, range(FIXED_SIZE)))

    df["relative_e1_pos_without_indicator"] = df["e1_pos_without_indicator"].map(
        lambda x: map(lambda y: FIXED_SIZE - 1 + x - y, range(FIXED_SIZE)))
    df["relative_e2_pos_without_indicator"] = df["e2_pos_without_indicator"].map(
        lambda x: map(lambda y: FIXED_SIZE - 1 + x - y, range(FIXED_SIZE)))

    os.chdir("/home/zy/data/wordvec")

    wiki_words, wiki_lower_words, wiki_vec = read_vec("wiki_fasttext.json", "wiki_fasttext.npy")
    all_words = set()
    for i in range(len(df)):
        words = set(df["words_with_indicator"][i])
        all_words = all_words.union(words)

    my_words = {}
    my_lower_words = {}
    my_vec = []
    index = 0

    for word in all_words:
        if word in wiki_words:
            my_vec.append(wiki_vec[wiki_words[word]])
            my_words[word] = index
            my_lower_words[word.lower()] = index
            index += 1
        else:
            word = word.lower()
            if word in wiki_lower_words:
                my_vec.append(wiki_vec[wiki_lower_words[word]])
                my_lower_words[word] = index
                index += 1
    my_vec = np.array(my_vec)

    df["wiki_index_with_indicator"] = df["words_with_indicator"].map \
        (lambda x: map(lambda y: get_word_index(y, my_words, my_lower_words), x))
    df["wiki_index_without_indicator"] = df["words_without_indicator"].map \
        (lambda x: map(lambda y: get_word_index(y, my_words, my_lower_words), x))

    df_train = df[:TRAIN_LEN]
    df_test = df[TRAIN_LEN:]

    train_wiki_with_indicator_index = np.array(df_train["wiki_index_with_indicator"].tolist())
    train_wiki_without_indicator_index = np.array(df_train["wiki_index_without_indicator"].tolist())
    test_wiki_with_indicator_index = np.array(df_test["wiki_index_with_indicator"].tolist())
    test_wiki_without_indicator_index = np.array(df_test["wiki_index_without_indicator"].tolist())

    train_relative_e1_pos_with_indicator = np.array(df_train["relative_e1_pos_with_indicator"].tolist())
    train_relative_e2_pos_with_indicator = np.array(df_train["relative_e2_pos_with_indicator"].tolist())
    train_relative_e1_pos_without_indicator = np.array(df_train["relative_e1_pos_without_indicator"].tolist())
    train_relative_e2_pos_without_indicator = np.array(df_train["relative_e2_pos_without_indicator"].tolist())

    test_relative_e1_pos_with_indicator = np.array(df_test["relative_e1_pos_with_indicator"].tolist())
    test_relative_e2_pos_with_indicator = np.array(df_test["relative_e2_pos_with_indicator"].tolist())
    test_relative_e1_pos_without_indicator = np.array(df_test["relative_e1_pos_without_indicator"].tolist())
    test_relative_e2_pos_without_indicator = np.array(df_test["relative_e2_pos_without_indicator"].tolist())

    train_e1_pos_with_indicator = np.array(df_train["e1_pos_with_indicator"].tolist())
    train_e2_pos_with_indicator = np.array(df_train["e2_pos_with_indicator"].tolist())
    train_e1_pos_without_indicator = np.array(df_train["e1_pos_without_indicator"].tolist())
    train_e2_pos_without_indicator = np.array(df_train["e2_pos_without_indicator"].tolist())

    test_e1_pos_with_indicator = np.array(df_test["e1_pos_with_indicator"].tolist())
    test_e2_pos_with_indicator = np.array(df_test["e2_pos_with_indicator"].tolist())
    test_e1_pos_without_indicator = np.array(df_test["e1_pos_without_indicator"].tolist())
    test_e2_pos_without_indicator = np.array(df_test["e2_pos_without_indicator"].tolist())

    train_labels = np.array(df_train["label"].tolist())
    test_labels = np.array(df_test["label"].tolist())

    os.chdir("/home/zy/data/zy_paper/task8")
    np.save("train_wiki_with_indicator_index.npy", train_wiki_with_indicator_index)
    np.save("train_wiki_without_indicator_index.npy", train_wiki_without_indicator_index)
    np.save("test_wiki_with_indicator_index.npy", test_wiki_with_indicator_index)
    np.save("test_wiki_without_indicator_index.npy", test_wiki_without_indicator_index)

    np.save("train_relative_e1_pos_with_indicator.npy", train_relative_e1_pos_with_indicator)
    np.save("train_relative_e2_pos_with_indicator.npy", train_relative_e2_pos_with_indicator)
    np.save("train_relative_e1_pos_without_indicator.npy", train_relative_e1_pos_without_indicator)
    np.save("train_relative_e2_pos_without_indicator.npy", train_relative_e2_pos_without_indicator)

    np.save("test_relative_e1_pos_with_indicator.npy", test_relative_e1_pos_with_indicator)
    np.save("test_relative_e2_pos_with_indicator.npy", test_relative_e2_pos_with_indicator)
    np.save("test_relative_e1_pos_without_indicator.npy", test_relative_e1_pos_without_indicator)
    np.save("test_relative_e2_pos_without_indicator.npy", test_relative_e2_pos_without_indicator)

    np.save("train_e1_pos_with_indicator.npy", train_e1_pos_with_indicator)
    np.save("train_e2_pos_with_indicator.npy", train_e2_pos_with_indicator)
    np.save("train_e1_pos_without_indicator.npy", train_e1_pos_without_indicator)
    np.save("train_e2_pos_without_indicator.npy", train_e2_pos_without_indicator)

    np.save("test_e1_pos_with_indicator.npy", test_e1_pos_with_indicator)
    np.save("test_e2_pos_with_indicator.npy", test_e2_pos_with_indicator)
    np.save("test_e1_pos_without_indicator.npy", test_e1_pos_without_indicator)
    np.save("test_e2_pos_without_indicator.npy", test_e2_pos_without_indicator)

    np.save("train_labels.npy", train_labels)
    np.save("test_labels.npy", test_labels)

    np.save("wiki_vec.npy", my_vec)

    np.save("y_sent_Dep_50.npy", np.array(df["label"].tolist()))
