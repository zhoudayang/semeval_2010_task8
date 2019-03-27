import re


def reverse_relation(relation):
    if relation[-9:] == "(e2,e1)\r\n":
        return relation[:-9] + "(e1,e2)\r\n"
    elif relation == "Other\r\n":
        return relation
    return relation[:-9] + "(e2,e1)\r\n"


def replace_e1(sentence, e2):
    result, _ = re.subn(r"<e1>(.+?)</e1>", "<e1>%s</e1>" % e2, sentence)
    return result


def replace_e2(sentence, e1):
    result, _ = re.subn(r"<e2>(.+?)</e2>", "<e2>%s</e2>" % e1, sentence)
    return result


def get_e1(sentence):
    e1_list = re.findall(r"<e1>(.+?)</e1>", sentence)
    assert (len(e1_list) == 1)
    return e1_list[0]


def get_e2(sentence):
    e2_list = re.findall(r"<e2>(.+?)</e2>", sentence)
    assert (len(e2_list) == 1)
    return e2_list[0]


if __name__ == "__main__":
    TRAIN_FILE = "/Users/zhouyang/datasets/SemEval2010_task8_all_data/" \
                 "SemEval2010_task8_training/TRAIN_FILE.TXT"

    lines = open(TRAIN_FILE).readlines()

    sentences = lines[0::4]
    relations = lines[1::4]
    second_lines = lines[2::4]
    third_lines = lines[3::4]

    reverse_relations = map(lambda x: reverse_relation(x), relations)

    e1s = map(lambda x: get_e1(x), sentences)
    e2s = map(lambda x: get_e2(x), sentences)

    reverse_lines = map(lambda x: replace_e1(x[0], x[1]), zip(sentences, e2s))
    reverse_lines = map(lambda x: replace_e2(x[0], x[1]), zip(reverse_lines, e1s))

    out_file = open("/Users/zhouyang/datasets/SemEval2010_task8_all_data/" \
                    "SemEval2010_task8_training/TRAIN_ALL_FILE.TXT", "wb")

    for line in lines:
        out_file.write(line)

    for i in range(len(reverse_lines)):
        out_file.write(reverse_lines[i])
        out_file.write(reverse_relations[i])
        out_file.write(second_lines[i])
        out_file.write(third_lines[i])
