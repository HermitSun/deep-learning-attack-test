## 深度学习对抗样本实验记录

### 个人感受

#### 模型训练

我觉得想要进行对抗样本生成，首先需要了解深度学习是什么，并且做过一些实验亲身体会过了，才能有的放矢。~~后来发现其实没什么用。~~

##### 全连接网络

训练数据的shape为(60000, 28, 28, 1)，基于训练数据的格式训练一个简单的全连接网络：

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

模型在测试集上的准确率是0.9173。

##### 预训练的CNN网络

为了尝试预训练网络，借用了赵文祺同学的GPU；在此表示诚挚的感谢。

但是这个问题太“简单”了，使用预训练网络的代价很大，但结果并不好，而且很多大型的网络并不支持对这种小问题的处理，比如著名的VGG16要求输入至少是32\*32的RGB图片，而原始数据集是28\*28的灰度图片，因此没有使用。

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
  callbacks_list.append(LearningRateScheduler(step_decay))
  ```

  但在这里效果也并不好，应该和Dropout效果不好的原因差不多；并且学习率优化需要和SGD优化器结合使用。

- 使用正则化效果也不好，无论是l1、l2还是l1_l2正则化效果都不好；最后还是选择了最简单的拓扑结构。