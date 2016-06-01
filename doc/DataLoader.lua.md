# DataLoader
该类用来从硬盘读取数据，并且给出1个batch的数据，送到nn里训练。

## __init()

###形参
opt
- h5_file, storytelling.h5文件的路径
- json_file，storytelling.json文件的路径

###成员变量

- self.ix_to_word, table类型，单词表。
 - key：单词编码，从1开始
 - value：单词的字符串
- self.vocab_size：9770，单词表大小
- image_size：table类型，h5文件中，image的维数，{247855, 3, 256, 256}
- self.num_images，247855。全部图像是80124个，为了提高读取效率，h5文件中的图像的个数是247855，和story的所有的标注是一一对应。
- self.num_images, 247855
- self.num_channels，3
- self.max_image_size, 256
- self.seq_length, 16, 每个标注的长度
- self.label_start_ix, table类型，{1,6,11，...}，每个story的第一个标注在所有标注中的位置。注：一共247855个标注。

###索引和迭代器
split_ix：table型，字典。标志出story列表中，哪些分别属于训练急、测试集和验证集。
- train：值是table，每个元素是story的列表中，属于train的story的下标。
- test：值是table，每个元素是story的列表中，属于test的story的下标。
- val：值是table，每个元素是story的列表中，属于val的story的下标。

iterators：table型。下一个要读取的batch的在split_ix中的序号。
- train：下一个要读取的训练集的batch的位置。
- test：下一个要读取的测试集的batch的位置。
- val：下一个要读取的验证集的batch的位置。

## DataLoader:resetIterator(split)

split：数据集的类型，train, test或者val。
重置split的迭代器（计数器）

## DataLoader:getVocabSize()

返回self.ix_size。9770.

## DataLoader:getVocab()

返回self.ix_to_word，单词表

## DataLoader:getSeqLength()

返回self.seq_length，每个图像的标注的长度，16

## DataLoader:getBatch(opt)

### 形参
opt
- batch_size, 10
- split, 'train','test'或者'val', 数据集的类型
- images_per_story, 5

### 成员变量

- split，数据集的类型，假设是'train'
- batch_size, 10
- images_per_story, 5
- split_ix, table类型，story列表中，属于’train‘类型的story的下标的table。
- max_index, 'train’类型的story的个数，39646
- wrapped, ???????

story_batch, table类型，列表，有batch_size(10)个元素，每个元素是一个story。stroy_batch[1]的结构是：
- images, 4D tensor，(images_per_story, 3, 256, 256), (5, 3, 256, 256), 5个图像
- labels, 2D tendor, (images_per_story, self.seq_length), (5, 16)，5个标注

### 返回值
data，table类型
data.images, table类型，列表，第t个元素是 story 的第t个图像， 每个元素是 images tensor，(batch_size, 3, 256, 256)， (10, 3, 256, 256)
data.labels，table类型，列表，第t个元素是story的第t个标注， 每个元素是 tensor，(batch_size， seq_length), (10, 16)
