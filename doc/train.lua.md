# train.lua

## 运行输入参数
- input_h5, 'storytelling.h5', 需要处理的h5 文件
- input_json, 'storytelling.json'，需要处理的json文件
- cnn_proto, 'model/VGG_ILSVRC_16_layers_deploy.prototxt'
- cnn_model, 'model/VGG_ILSVRC_16_layers.caffemodel'
- rnn_size, 512
- input_encoding_size, 512
- max_iters, -1
- batch_size, 10
- images_per_story, 5，每个story包含的图像的数量
- finetune_cnn_after, -1
- grad_clip, 0.1
- losses_log_every, 30
- checkpoint_path, 
- id
- gpuid, 0
- backend, 'cudnn'
- seed, 123

##初始化网络
lmOpt, 语言模型的相关参数
- vocab_size, 9770
- input_encoding_size, 512
- rnn_size, 512
- num_layers, 1，rnn 网络的隐藏层层数
- seq_length, 16，每个story的每个图像的标注的长度是16
- batch_size, 10