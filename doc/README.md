# storytelling 架构
[TOC]
## 概述

文件夹：dataset, doc
文件：prepro.py, README.md

## SIND数据集

SIND数据集的任务在于根据多幅图像，生成一段高级语义的话。SIND的图像来源于flickr，大约有200k。由于图像多，像素高，所以SIND没有提供打包好的图像数据集，仅仅提供了图像的URLs。我用wget命令手动下载这些图像，发现很多链接都失效了。通过清洗SIND数据集，我生成了新的训练集、测试集和验证集，以及标注。

- 数据集链接：[http://sind.ai](http://sind.ai)
- arxiv论文：[http://arxiv.org/abs/1604.03968](http://arxiv.org/abs/1604.03968)

## dataset文件夹
包括：

 - train，训练集，64104个图像
 - test，测试集，8085个图像
 - val，验证集，7935个图像
 - DII，descriptions of images-in-isolation，json格式
 - SIS，stories of images-in-sequence，json格式
 - sind_preprocess.py，对SIND数据进行预处理

###sind_preprocess.py

读取SIS文件夹下面的3个json文件，生成image_set.json和story_set.json文件

**image_set.json：** 字典，key: value。共209651个元素。即，SIND数据集共包含了209651个图像。

- key，图像的id。
- value，字典，图像的详细信息
 - imagename，图像文件名
 - split，train|test|val，表示图像属于属于什么类型
 - anno_num，引用到该图像的标注的次数

**story_set.json：**字典，key: value。共49571个元素。即，经过清洗后，共有49571个stories

- key，story的id
- value，字典，story的详细信息
 - image_id：列表，5个图像的id
 - imagename：列表，5个图像的文件名
 - text：列表，5条标注信息，全部是小写。
 - img_num：该story的标注的图像数量，恒为5。

统计信息：

- SIND数据集共有 209651 个图像
 - train:167528
 - test:21075
 - val:21048
- 去除无法下载和没有被使用的图像，共有 80124 个图像
 - train:64104
 - test:8085
 - val:7935
- story number: 49571
 - train:39646
 - test:4995
 - val:4930
- annotation number：247855
 - train：198230
 - test:24975
 - val:24650

## prepo.py

prepro.py的任务：

- 生成单词表
- 对标注进行编码
- 把图片resize成256x256大小，存成hdf5格式

参数设置：

- max_length：16，每幅图像的标注，最长是16，超过部分截断。
- word_count_threshold：5，出现次数小于5的单词，从标注和单词表中删除，用“UNK”表示，加入到单词表中。

统计信息：

- total words: 2516809
- number of bad words: 20235/30004 = 67.44%
- number of UNKs: 36191/2516809 = 1.44%，出现次数小于5的单词是UNK。
- max length sentence in raw data:  82
- max length story in raw data:  247
- story number: 49571
- number of words in vocab: 9770，包括UNK，不包括标点。

*输出文件：**storytelling.h5, storytelling.json***
storytelling.h5:
- labels: 列表，有247855个元素。每个元素是一个长度16的整型数组，是标注的编码。
- label_start_ix：列表，有49561个元素。第k个值，表示第k个story的标注在“labels”中的位置。
- images：4D tensor，存resize之后的图片。(247855,3,256,256)

storytelling.json，字典：
- ix_to_word：字典，单词表，大小为9770。
 - key：单词数字编码
 - value：单词字符串
- story：列表，49671个元素，每个元素是story的详细信息。列表的顺序和storytelling.h5的labels和label_start_ix的顺序是一致的。

