# coding=utf-8
# 使用100维的wiki训练的语料进行处理

import json
import numpy as np
import os

np.random.seed(1337)

if __name__ == "__main__":
    os.chdir("/home/zy/data/wordvec")

    embedding = np.load("google.npy")
    words = json.load(open("google_list.json"))

    os.chdir("/home/zy/data/zy_paper/common_pretreat")

    all_words = json.load(open("word_list.json"))

    # wiki_embedding = np.zeros(len(all_words), dim=100)
    num_words = len(all_words)
    dim = embedding.shape[1]
    google_embedding = np.random.uniform(-0.5, 0.5, size=(num_words, dim))

    oov_cnt = 0

    for word in all_words:
        if word in words:
            google_embedding[all_words[word]] = embedding[words[word]]
        else:
            oov_cnt += 1

    np.save("google_vec.npy", google_embedding)
    print oov_cnt