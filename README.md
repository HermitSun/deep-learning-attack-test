## 深度学习对抗样本实验记录

我觉得想要进行对抗样本生成，首先需要了解深度学习是什么，并且做过一些实验亲身体会过了，才能有的放矢。

### 训练基准模型

#### 拓扑与性能

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

基准模型在测试集上的准确率是0.8871。

#### 训练过程与选择

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
  sgd = optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
  ```

  基于轮数的学习率衰减：

  ```python
  def step_decay(epoch):
      initial_l_rate = 0.1
      drop = 0.5
      epochs_drop = 10.0
      l_rate = initial_l_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
      return l_rate
  callbacks_list.append(LearningRateScheduler(step_decay))
  ```

  但在这里效果也并不好，应该和Dropout效果不好的原因差不多；并且学习率优化需要和SGD优化器结合使用。

- 使用正则化效果也不好，无论是l1、l2还是l1_l2正则化效果都不好；最后还是选择了最简单的拓扑结构。

