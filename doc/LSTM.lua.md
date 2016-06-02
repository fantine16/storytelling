# LSTM

## 形参
- input_size, 512
- output_size, 9771, 包括START 和 END TOKENS，vocab_size + 1
- rnn_size, 512
- n, 1, 隐层个数
- dropout, 0.5

## 初始条件

假设 batch_size=10

## LSTM的输入

inputs={}，包括3部分
- 单词或者图像的编码，(10,512)
- prev_c, (10, 512)
- prev_h, (10, 512)

## LSTM的输出

outputs={}，包括3部分
- next_c, (10, 512)
- next_h, (10, 512)
- 单词的概率, (10, 9771), 行向量表示预测的所有的单词的概率