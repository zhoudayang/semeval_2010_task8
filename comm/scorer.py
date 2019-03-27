# coding:utf-8
import numpy as np
import sys
import os

sys.path.append("..")
from pretreat.semeval_2010 import relations


def write_results(predictions, result_file="/home/zy/git/zy_paper/comm/temp_result.txt"):
    start_no = 8001
    id_to_relation = dict(zip(relations.values(), relations.keys()))
    with open(result_file, "w") as file:
        for idx, id in enumerate(predictions):
            rel = id_to_relation[id]
            file.write('%d\t%s\n' % (start_no + idx, rel))


def convert_labels(labels):
    labels = list(labels)
    labels = map(lambda x: x.argmax(), labels)
    return labels


def show_result(result):
    begin = result.rfind("MACRO-averaged result (excluding Other)")
    end = result.rfind("\n\n\n\n")
    return result[begin: end]


def get_marco_f1(predict_labels, real_labels):
    if isinstance(predict_labels, np.ndarray) and len(predict_labels.shape) == 2:
        predict_labels = convert_labels(predict_labels)
    # if isinstance(real_labels, np.ndarray) and len(real_labels.shape) == 2:
    #     real_labels = convert_labels(real_labels)
    write_results(predict_labels)
    os.chdir("/home/zy/git/zy_paper/comm")
    command = "perl scorer.pl test_keys.txt temp_result.txt"
    result = os.popen(command).read()
    print show_result(result)
    begin_index = result.find("macro-averaged F1")
    result = result[begin_index + 19:-6]
    return float(result)


if __name__ == "__main__":
    pass
