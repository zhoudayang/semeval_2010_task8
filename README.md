# Remark for semeval 2010 task8

prework: 当前实现的最好效果为cnn + piecewise max pooling 83.6

10.13
在实现基准模型，cnn + max pooling + softmax 时发现，f1最好效果只有63
将cnn激活函数更换为relu，效果提升到71

10.15
上述结果有误，调整网络结构，发现max pooling实现有误，更换之后发现f1最好效果提升到82.58， 相差0.21
使用新版本的wiki词向量，发现此词向量的效果不如google news，f1效果为81.50
todo: 使用window=5版本的词向量试一下
todo: 为未登录词使用同一个词向量

10.16
开始建立新模型： TCA-CNN
??? ranking loss function 存在问题

10.17
调整了position embedding的输入部分，发现simple_cnn的最好效果达到了83.09
todo: redo -> lstm + attention
todo: how to use relation embedding and ranking loss function
todo: check the performance of pytorch implementation of multi-level cnn

10.22
加入pos信息,f1最好效果达到83.82
发现采用数据增强,实际运行效果大跌,猜测由于网络结构过于扁平,导致学习失败.

10.23
重新进行了预处理,统一设置word index, 便于更换word embedding
测试了lstm_with_attention的性能,实际最好效果74
测试双向lstm的基准性能,用于和lstm + attention进行比较, 实际效果为: 0.645左右
更换google词向量,发现最好效果为0.6418
??猜测预处理有问题
simple_cnn 最好效果83.20  发现使用tensorflow, 无法得到可复现的结果
处理: 暂时不纠结预处理过程, 在google embedding的基础上继续处理
picewise-maxpooling 无效, 实际效果在250 * 4 cnn作用下,可以达到0.83, 不如直接采用maxpooling

11.27 
bert 大法好，实际效果不过10，醉了~

12.6
rnf_cnn + elmo 84.5
todo: add pos_tag


