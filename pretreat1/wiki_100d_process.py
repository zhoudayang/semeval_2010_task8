# coding=utf-8
# 使用100维的wiki训练的语料进行处理

import json
import numpy as np
import os

np.random.seed(1337)

if __name__ == "__main__":
    os.chdir("/home/zy/data/wiki_win10_100d_20180420")

    embedding = np.load("wiki.vec.npy")
    words = json.load(open("words.json"))

    os.chdir("/home/zy/data/zy_paper/common_pretreat")

    all_words = json.load(open("word_list.json"))

    # wiki_embedding = np.zeros(len(all_words), dim=100)
    num_words = len(all_words)
    dim = embedding.shape[1]
    wiki_embedding = np.random.uniform(-0.5, 0.5, size=(num_words, dim))

    oov_cnt = 0

    for word in all_words:
        if word in words:
            wiki_embedding[all_words[word]] = embedding[words[word]]
        else:
            oov_cnt += 1

    np.save("wiki_100d_vec.npy", wiki_embedding)