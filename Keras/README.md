# Keras 例子
addition_rnn.py 用RNN拟合加法运算

antirectifier.py 为Keras编写自定义图层

babi_memnn.py 在[bAbI](https://research.facebook.com/research/babi/)数据集上训练记忆网络以进行阅读理解

babi_rnn.py 在bAbI数据集上训练两分支递归网络以进行阅读理解

cifar10_cnn.py 在CIFAR10小图像数据集上训练简单CNN

conv_filter_visualization.py 通过梯度上升可视化VGG16的过滤器

deep_dream.py 为Keras编写深梦

image_ocr.py 训练卷积、循环和CTC logloss函数以执行OCR识别

imdb_bidirectional_lstm.py 在IMDB情绪分类任务上训练双向LSTM

imdb_cnn.py 使用Convolution1D进行文本分类

imdb_cnn_lstm.py 在IMDB情绪分类任务上训练卷积、循环网络

imdb_fasttext.py 在IMDB情绪分类任务上训练FastText模型

imdb_lstm.py 在IMDB情绪分类任务上训练LSTM

lstm_benchmark.py 在IMDB情绪分类任务上比较不同的LSTM实现

lstm_text_generation.py 从尼采的书中生成文本

mnist_cnn.py 在MNIST数据集上训练简单的ConvNet

mnist_hierarchical_rnn.py 训练分层RNN以对MNIST数字进行分类

mnist_irnn.py 使用MNIST数据集的IRNN实验

mnist_mlp.py 在MNIST数据集上训练简单的多层感知器

mnist_net2net.py 使用MNIST再现Net2Net实验

mnist_siamese_graph.py 使用MNIST数据集训练一个多层感知器

mnist_sklearn_wrapper.py 使用sklearn包装器

mnist_swwae.py 使用MNIST数据集训练自动编码器

mnist_transfer_cnn.py 转移学习

neural_doodle.py 神经涂鸦

neural_style_transfer.py 艺术画生成

pretrained_word_embeddings.py 在新闻组数据集上训练文本分类模型

reuters_mlp.py 在路透社newswire主题分类任务上训练和评估简单的MLP

stateful_lstm.py 使用有状态的RNN来有效地建模长序列

variational_autoencoder.py 构建变分自动编码器

variational_autoencoder_deconv.py 使用解卷积层与Keras构建变分自动编码器


# Keras examples

[addition_rnn.py](addition_rnn.py)
Implementation of sequence to sequence learning for performing addition of two numbers (as strings).

[antirectifier.py](antirectifier.py)
Demonstrates how to write custom layers for Keras.

[babi_memnn.py](babi_memnn.py)
Trains a memory network on the bAbI dataset for reading comprehension.

[babi_rnn.py](babi_rnn.py)
Trains a two-branch recurrent network on the bAbI dataset for reading comprehension.

[cifar10_cnn.py](cifar10_cnn.py)
Trains a simple deep CNN on the CIFAR10 small images dataset.

[conv_filter_visualization.py](conv_filter_visualization.py)
Visualization of the filters of VGG16, via gradient ascent in input space.

[deep_dream.py](deep_dream.py)
Deep Dreams in Keras.

[image_ocr.py](image_ocr.py)
Trains a convolutional stack followed by a recurrent stack and a CTC logloss function to perform optical character recognition (OCR).

[imdb_bidirectional_lstm.py](imdb_bidirectional_lstm.py)
Trains a Bidirectional LSTM on the IMDB sentiment classification task.

[imdb_cnn.py](imdb_cnn.py)
Demonstrates the use of Convolution1D for text classification.

[imdb_cnn_lstm.py](imdb_cnn_lstm.py)
Trains a convolutional stack followed by a recurrent stack network on the IMDB sentiment classification task.

[imdb_fasttext.py](imdb_fasttext.py)
Trains a FastText model on the IMDB sentiment classification task.

[imdb_lstm.py](imdb_lstm.py)
Trains a LSTM on the IMDB sentiment classification task.

[lstm_benchmark.py](lstm_benchmark.py)
Compares different LSTM implementations on the IMDB sentiment classification task.

[lstm_text_generation.py](lstm_text_generation.py)
Generates text from Nietzsche's writings.

[mnist_cnn.py](mnist_cnn.py)
Trains a simple convnet on the MNIST dataset.

[mnist_hierarchical_rnn.py](mnist_hierarchical_rnn.py)
Trains a Hierarchical RNN (HRNN) to classify MNIST digits.

[mnist_irnn.py](mnist_irnn.py)
Reproduction of the IRNN experiment with pixel-by-pixel sequential MNIST in "A Simple Way to Initialize Recurrent Networks of Rectified Linear Units" by Le et al.

[mnist_mlp.py](mnist_mlp.py)
Trains a simple deep multi-layer perceptron on the MNIST dataset.

[mnist_net2net.py](mnist_net2net.py)
Reproduction of the Net2Net experiment with MNIST in "Net2Net: Accelerating Learning via Knowledge Transfer".

[mnist_siamese_graph.py](mnist_siamese_graph.py)
Trains a Siamese multi-layer perceptron on pairs of digits from the MNIST dataset.

[mnist_sklearn_wrapper.py](mnist_sklearn_wrapper.py)
Demonstrates how to use the sklearn wrapper.

[mnist_swwae.py](mnist_swwae.py)
Trains a Stacked What-Where AutoEncoder built on residual blocks on the MNIST dataset.

[mnist_transfer_cnn.py](mnist_transfer_cnn.py)
Transfer learning toy example.

[neural_doodle.py](neural_doodle.py)
Neural doodle.

[neural_style_transfer.py](neural_style_transfer.py)
Neural style transfer.

[pretrained_word_embeddings.py](pretrained_word_embeddings.py)
Loads pre-trained word embeddings (GloVe embeddings) into a frozen Keras Embedding layer, and uses it to train a text classification model on the 20 Newsgroup dataset.

[reuters_mlp.py](reuters_mlp.py)
Trains and evaluate a simple MLP on the Reuters newswire topic classification task.

[stateful_lstm.py](stateful_lstm.py)
Demonstrates how to use stateful RNNs to model long sequences efficiently.

[variational_autoencoder.py](variational_autoencoder.py)
Demonstrates how to build a variational autoencoder.

[variational_autoencoder_deconv.py](variational_autoencoder_deconv.py)
Demonstrates how to build a variational autoencoder with Keras using deconvolution layers.
