#storytelling 架构

##概述
文件夹：dataset, doc
文件：prepro.py, README.md


##SIND数据集
- 数据集链接：[http://sind.ai](http://sind.ai)
- arxiv论文：[http://arxiv.org/abs/1604.03968](http://arxiv.org/abs/1604.03968)

SIND数据集的任务在于根据多幅图像，生成一段高级语义的话。SIND的图像来源于flickr，大约有200k。由于图像多，像素高，所以SIND没有提供打包好的图像数据集，仅仅提供了图像的URLs。我用wget命令手动下载这些图像，发现很多链接都失效了。通过清洗SIND数据集，我生成了新的训练集、测试集和验证集，以及标注。


##dataset文件夹
包括：
- train，训练集，64104个图像
- test，测试集，8085个图像
- val，验证集，7935个图像
- DII，descriptions of images-in-isolation，json格式
- SIS，stories of images-in-sequence，json格式
- sind_preprocess.py，对SIND数据进行预处理

###sind_preprocess.py
