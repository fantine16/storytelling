# LanguageModel类

语言模型，新建了LanguageModel类，重写了updateOutput()方法和updateGradInput()方法。实现story的LSTM前向和反向传播。

```lua
local layer, parent = torch.class('nn.LanguageModel', 'nn.Module')
```
## layer:__init(opt)
- self.vocab_size, 9770, 单词表大小
- self.input_encoding_size, 图像和单词的输入的编码长度
- self.rnn_size， 512，rnn隐层的节点的数量
- self.num_layers，1，rnn的隐层的数量
- self.images_per_story，5
- dropout, 0.5
- self.seq_length, 16


- self.core, LSTM结构
```lua
LSTM.lstm(self.input_encoding_size, self.vocab_size + 1, self.rnn_size, self.num_layers, dropout)
```
LSTM.lstm(512, 9771, 512, 1, 0.5)

- self.lookup_table
```lua
 nn.LookupTable(self.vocab_size + 1, self.input_encoding_size)
```
nn.LookupTable(9771, 512)

## layer:updateOutput(input)

重写了语言模型的updateOutput()方法。forward()自动调用这个方法。


input: 1个batch的数据， table类型
- imgs, imgs的第t个元素是story的第t个图像，每个元素是 tensor，(10, 3, 224, 224)
- labels,  labels的第t个元素是story的第t个标注， 每个元素是 tensor，(10, 16)
- batch_size, 10
- seq_length, 16

### 返回值
self.output, tensor 类型
((self.seq_length+2)*self.images_per_story, batch_size, self.vocab_size+1)
(90, 10, 9771)
每个story的标注的长度是90，batch_size是10，算上start token和end token，单词表大小是9771. start token 和 end token用同一个符号单词表示。

# LanguageModelCriterion类

根据loss function 求梯度

## 输入参数

input
1个batch的story的5个标注的所有单词的概率。tensor, (90, 10, 9771)

labels
1个batch的story的真实的标注。

## 局部变量

- L, 90
- N, 10, batch的大小
- Mp1, 9771, 表示 start token 和 end token
- num_img_per_story, 5
- seq_length, 16

## 返回值

loss