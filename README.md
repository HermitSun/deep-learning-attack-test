## 深度学习对抗样本实验记录

### 项目结构

#### 项目介绍

TODO

#### 代码结构

```powershell
│  generate.py
│  model.hdf5
│  README.pdf
│
├─attack_data
│      attack_data.npy
│
├─models
│      base_model.hdf5
│      base_model_train.py
│      cnn_model.hdf5
│      cnn_model_train.py
│      model_test.py
│
├─process
│      split_data.py
│      ssim.py
│
└─test_data
        test_data.npy
```

#### 运行方法

加载模型，传入shape为(n, 28, 28, 1)的ndarray（n为正整数），然后调用模型的`evaluate`方法。

示例代码：

```python
model = load_model("model.hdf5")
test_data = test_data.reshape(sample_num, width, height, channels)
val_loss, val_acc = model.evaluate(test_data, test_labels)
print("Model Accuracy:", val_acc)
```

更详细的使用方法参见`models/model_test.py`。

#### 生成时间

TODO

### 算法详解

TODO

### 个人感受

#### 模型训练

我觉得想要进行对抗样本生成，首先需要了解深度学习是什么，并且做过一些实验亲身体会过了，才能有的放矢。~~后来发现重点是样本生成，训练模型其实没什么用，我还傻乎乎地炼了三天丹。~~

##### 全连接网络

训练数据的shape为(60000, 28, 28, 1)，基于训练数据的格式训练一个简单的全连接网络。网络拓扑如下：

```python
def create_base_model():
    model = models.Sequential()
    model.add(layers.Dense(28 * 28, activation="relu", input_shape=(28 * 28,)))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["acc"])
    return model
```

模型在测试集上的准确率是0.8871。

##### 卷积神经网络

因为这是一个计算机视觉问题，卷积神经网络相比全连接网络能保留原始图片的空间结构；并且，CNN具有很多适合解决视觉问题的特性。

因为图片有三个维度，width、height、channel，全连接网络相当于是直接把图片“拉平”了，变成了一个维度；而CNN能保留这种空间上的特性。同时，因为视觉世界从根本上具有平移不变性和空间层次结构，而CNN学到的模式符合这些特性[1]。

我觉得，人认识视觉世界的方式也是从局部到整体，往往是局部细节的组合让人对事物产生了认知。CNN学习的是局部模式，相对来说也更符合人的认知过程。

因为输入数据较小，卷积层不能太多，否则会溢出。网络拓扑如下：

```python
def create_cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation="relu", input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(28 * 28, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    return model
```

模型在测试集上的准确率是0.9176。~~但是再次运行只剩0.9149，也不知道为什么。~~

##### 预训练的CNN网络

为了尝试预训练网络，借用了赵文祺同学的GPU；在此表示诚挚的感谢。

但是这个问题太“简单”了，使用预训练网络的代价很大，但效果并不好，而且很多大型的网络并不支持对这种小问题的处理，比如著名的VGG16要求输入至少是32\*32的RGB图片，而原始数据集是28\*28的灰度图片，因此没有使用。

##### 深度可分离卷积

即depthwise separable convolution。

实际使用效果不是很好，因此没有使用。猜测是因为输入数据的空间位置不太相关，并且是单通道的灰度图片，不存在通道间的独立，导致效果欠佳。

##### 训练过程与选择

- 因为数据量相对来说不算太小，出于简便和效率上的考虑，直接使用了留出验证，而没有使用更复杂也更可靠的K折验证。

- 在优化器的选择上，RMSProp与Adam性能相近；这里选择了Adam。

- 为了防止模型过拟合，增加了Dropout层，并按照原论文所述，使用了自定义的SGD优化器：

  ```python
  sgd = optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
  ```

  估计是因为模型简单，训练轮次不多，还没有到过拟合的时候，效果并不好，故舍弃。

- 此外还尝试使用了学习率优化。基于时间的学习率衰减：

  ```python
  epochs = 10
  learning_rate = 0.1
  decay_rate = learning_rate / epochs
  momentum = 0.8
  sgd = optimizers.SGD(
      lr=learning_rate, 
      momentum=momentum, 
      decay=decay_rate, 
      nesterov=False
  )
  ```

  基于轮数的学习率衰减：

  ```python
  def step_decay(epoch):
      initial_l_rate = 0.1
      drop = 0.5
      epochs_drop = 10.0
      l_rate = initial_l_rate * math.pow(
          drop, math.floor((1 + epoch) / epochs_drop)
      )
      return l_rate
  callbacks_list.append(
      LearningRateScheduler(step_decay)
  )
  ```

  但在这里效果也并不好，应该和Dropout效果不好的原因差不多；并且学习率优化需要和SGD优化器结合使用。

- 使用正则化效果也不好，无论是l1、l2还是l1_l2正则化效果都不好；最后还是选择了最简单的拓扑结构。

#### 杂项

- 针对这种计算机视觉的问题，还是得用卷积神经网络。

- 使用matplotlib进行学习过程的可视化，以调整epochs避免过拟合；这是传统的方法，先进行足够多的轮次，然后进行调整。但这个不是最好的方案，会浪费大量的计算资源。可以用keras回调或者TensorFlow提供的TensorBoard来更好地进行处理。

- 使用CNN时可以使用summary()来查看网络拓扑，以选择合适的input_shape或者input_dim；这一点在使用现成的预训练网络时特别好用，尤其是层数一多就算不清楚卷积层大小了。
- 使用CNN的时候要注意，尤其是针对小图片，要控制卷积的数量，否则可能会溢出。

- 生成器的使用；generator真的是很有意思的一个语言特性，类似于无限循环的一个闭包，并且是懒求值的。也许就是为了刻画离散数学里的闭包？

- CoLab是个好东西。
- 这个模型有点简单，特征提取、数据增强等等避免小型图像数据集过拟合的方式都没什么用武之地。
- 很奇怪的一件事是，在小规模样本上使用VGG16对数据增强后的图片进行训练的效果并不好，只有0.8左右的准确率。一开始以为是CPU进行浮点计算的精度不够，但换成GPU之后还是这个情况。为什么呢？

### 参考资料

[1] Chollet F. Deep Learning mit Python und Keras: Das Praxis-Handbuch vom Entwickler der Keras-Bibliothek[M]. MITP-Verlags GmbH & Co. KG, 2018. 

[2] Brownlee J. Deep learning with python: Develop deep learning models on theano and tensorflow using keras[M]. Machine Learning Mastery, 2016. 

[3] Srivastava N, Hinton G, Krizhevsky A, et al. Dropout: a simple way to prevent neural networks from overfitting[J]. The journal of machine learning research, 2014, 15(1): 1929-1958. 

[4] Kingma D P, Ba J. Adam: A method for stochastic optimization[J]. arXiv preprint arXiv:1412.6980, 2014. 

[5] Tieleman T, Hinton G. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude[J]. COURSERA: Neural networks for machine learning, 2012, 4(2): 26-31. 

[6]  Wang Z, Bovik A C, Sheikh H R, et al. Image quality assessment: from error visibility to structural similarity[J]. IEEE transactions on image processing, 2004, 13(4): 600-612. 

[7]  Xiao C, Li B, Zhu J Y, et al. Generating adversarial examples with adversarial networks[J]. arXiv preprint arXiv:1801.02610, 2018. 