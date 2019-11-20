## 深度学习对抗样本实验记录

### 项目结构

#### 项目介绍

项目使用基于FGSM改进的预训练的二次迭代FGSM with momentum + 联机的图像合成算法，在相对较短的时间内即可生成较为有效的对抗样本，在测试集上能达到0.8097的攻击成功率和0.9116的平均SSIM。

#### 代码结构

```powershell
│  generate.py                      # 可调用的对抗样本生成函数
│  model.hdf5                       # 基于fashion_mnist训练的模型
│  README.md                        # 本说明文件
│
├─attacks                           # 对抗样本相关
│      base_attack.py                   # 随机噪声
│      data_index_map.dat               # 测试集中数据和序号的对应关系
│      data_label_map.dat               # 测试集中数据和标签的对应关系
│      FGSM_with_momentum_attack.py     # 带动量的FGSM
│      iFGSM_attack.py                  # iFGSM
│      sample_composition_attack.py     # 图像合成
│
├─attack_data                       # 生成的对抗样本
│      attack_data.npy
│
├─models                            # 模型相关，包括模型的训练和测试过程
│      base_model.hdf5                  # 全连接模型
│      base_model_train.py
│      cnn_model.hdf5                   # CNN模型
│      cnn_model_train.py
│      model_test.py                    # 自己编写的模型测试
│
├─process                           # 整个项目的处理过程
│      concat_attack_data.py            # 合并对抗样本
│      get_data_index_map.data.py       # 生成map
│      noises.py                        # 生成噪声
│      split_data.py                    # 分离fashion_mnist的数据
│      ssim.py                          # SSIM计算
│
└─test_data                         # fashion_mnist的测试数据
       test_data.npy
```

#### 运行方法

加载模型和数据，调用`generate`方法，传入需要生成对抗样本的数据和数据的shape，然后用生成的对抗样本对模型进行攻击。

```python
# load model
model = load_model("model.hdf5")

# load test data and labels
(train_data, train_labels), (t_data, test_labels) = fashion_mnist.load_data()
test_data = np.load("test_data/test_data.npy")

# generate attack data
attack_data = generate(test_data, (len(test_data), 28, 28, 1))

# evaluate attack
attack_success(test_data, attack_data)
```

可运行的代码参见`example.py`，可以使用`python example.py`查看测试效果。因为数据量较大，计算SSIM需要一段时间，请稍作等待。

#### 生成时间

在单核CPU上，单次预训练（生成现有的对抗样本）花费32824.76  s。在实际使用中，CPU 8核并行计算，大约花费1.5 h。

#### 算法详解

预训练的二次迭代FGSM with momentum + 联机的图像合成。思路如下：

##### FGSM

即Fast Gradient Sign Method。在我的认知中，这个既属于增大损失攻击，又属于线性攻击。

![](https://i.loli.net/2019/11/18/rtsfgdqRxGkYIKm.png)

按照论文里的说法，这个方法的正确性在于：

1. 高维空间下的线性行为足以产生对抗样本[8]。同时，神经网络的激活函数大多是线性的，只要进行线性的扰动，就会产生很大的偏差。
2. 图片每一个通道只有255种可能的取值，表达能力有限；灰度图片只有1个通道，更是如此。
3. 模型训练的目的是减小损失函数的值，而为了做到这一点，往往是让权重的变化方向与梯度的变化方向相反。那么，只需要让变化量尽可能和梯度变化方向一致，即可增大损失，也就会对预测结果产生最大化的影响。sign函数保证了变化方法同梯度方向一致。对损失函数求偏导，即可得到与权值向量ω有关的函数。

##### iFGSM

因为一次线性变换效果不是可能特别显著，所以需要迭代地进行。这个过程也就是iFGSM，即迭代的FGSM。公式如下：

![](https://i.loli.net/2019/11/18/nmW2lRsTQzcqkHp.png)

##### FGSM with momentum

在迭代的基础上增加了动量。在我看来，可能是为了避免收敛到局部最优解，而是让结果尽可能靠近全局最优解，类似于物理上的动量。

![](https://camo.githubusercontent.com/923e92405256bbfebc25c1d6b6e44d6408e0fd87/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f675f253742742b312537442532302533442532302535436d7525323025354363646f74253230675f253742742537442532302b253230253543667261632537422535436e61626c615f253742782537444a253238785f253742742537442535452537422a253744253243792532392537442537422535432537432535436e61626c615f253742782537444a253238785f253742742537442535452537422a253744253243792532392535432537435f3125374425324325323025354371756164253230785f253742742b312537442535452537422a2537442532302533442532302535436d617468726d253742636c6970253744253238785f253742742537442535452537422a2537442532302b253230253543616c70686125354363646f742535436d617468726d2537427369676e253744253238675f253742742b31253744253239253239)

事实上，这个方法一般用于白盒攻击。但只要适当控制epochs，就可以使用该算法进行无目标的黑盒攻击。通过实验，我发现，绝大多数样本在10次迭代内就会使模型出现误判。所以只需将epochs置为10，就可以生成基本有效的对抗样本。

同时，这个算法存在一个缺陷（有可能是我实现的问题），有时候会因为迭代次数过多，像素点变为负数被忽略，导致返回原图，即两张图片的``SSIM = 1`；在这个时候需要将算法进行退化，对图片使用iFGSM再次进行处理，根据经验，此时一般进行5次迭代即可得到有效的对抗样本。

##### 图像合成

但是因为连模型都未知，所以FGSM只能用来预训练一批数据，在使用时进行查询。如果传入的数据查询不到，则需要现场生成；这时算法再次退化，退化为图像合成。该算法基于以下两个假设：

1. 与原始图像相似度越高的图片，越有可能使模型误判。
2. 与原始图像标签不同的图片，模型预测的结果也不同。

所以，该算法的核心是从原数据集里寻找与当前图片相似度最高且标签不同的图片。理论上，这个算法的应该在整个数据集中寻找相似度最高的图片，但是这样做的复杂度为O(n^2)，对于大规模数据集计算成本过高，不可接受；所以退而求其次，对于每张图片，从原数据集里随机选取k张图片，计算SSIM，选取SSIM最高的一张，与当前图片按比例进行合成。合成方法如下：

```python
attack = images[i] * alpha + images[j] * beta + gamma
```

此时，算法的复杂度降低到O(kn)，k为随机抽取的样本个数。

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

这个问题太“简单”了，使用预训练网络的代价很大，而且效果并不好。事实上，很多大型的网络并不支持对这种小问题的处理，比如著名的VGG16要求输入至少是32\*32的RGB图片，而原始数据集是28\*28的灰度图片，因此没有使用。

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

#### 算法选择

> 由于算力不足，有些暴力算法无法实现，并且直接导致有些算法的准确率下降，实在可惜。

在实践过程中，我发现，为了提高模型准确率，一般的手段包括：选用更合适的网络，使损失函数最小化，选择合适的激活函数。为了快速收敛，一般会让系数的变化向梯度的反方向传播；同时，激活函数具有线性性。我觉得这是对抗样本生成的突破口。我主要从这几个方面进行了尝试：随机攻击、增大损失攻击、线性攻击。

##### 随机噪声

这是最容易想到的方法，属于随机攻击的一种而且效果相当显著，但SSIM会很低，并且与机器学习基本没有关系。同时，由于生成噪声的成本过高（言下之意是性价比很低），在算力不足的情况下基本不予考虑。

###### 椒盐噪声

18%的随机噪声时，全连接模型的准确率下降到0.2，CNN仍有0.3的准确率。此时平均SSIM为0.3010。

23%的随机噪声时，两个模型的准确率均下降到0.2，但平均SSIM只有0.2295。

噪声比例继续增加，模型的准确率基本不变，SSIM不断下降。

###### 高斯噪声

高斯噪声对全连接网络的攻击效果特别好，但对CNN模型的攻击效果一般。

均值0.8，方差0.5时，全连接网络的准确率直接下降为0，但CNN网络仍有0.2的准确率；但此时平均SSIM只有0.1576。

##### 图像合成

参加“算法详解”。

##### FGSM

参见“算法详解”。

##### 单像素攻击

这是我的另一个备选算法，但因为差分进化的计算成本比较高，最终舍弃。

![](https://github.com/Hyperparticle/one-pixel-attack-keras/raw/deeaabfd2c75ba613c1c10d6d50c37555018568f/images/Ackley.gif)

##### advGAN

不采用生成对抗网络的一个主要原因和单像素攻击一致，算力不足。另一个方面，相对来说GAN不那么容易收敛，调参的时候有很多不确定因素。

#### 随感

- 搞机器学习得有钱，至少得有个GPU。

- 针对这种计算机视觉的问题，还是得用卷积神经网络。

- 使用matplotlib进行学习过程的可视化，以调整epochs避免过拟合；这是传统的方法，先进行足够多的轮次，然后进行调整。但这个不是最好的方案，会浪费大量的计算资源。可以用keras回调或者TensorFlow提供的TensorBoard来更好地进行处理。

- 使用CNN时可以使用summary()来查看网络拓扑，以选择合适的input_shape或者input_dim；这一点在使用现成的预训练网络时特别好用，尤其是层数一多就算不清楚卷积层大小了。

- 使用CNN的时候要注意，尤其是针对小图片，要控制卷积的数量，否则可能会溢出。

- 生成器的使用；generator真的是很有意思的一个语言特性，类似于无限循环的一个闭包，并且是懒求值的。也许就是为了刻画离散数学里的闭包？

- CoLab是个好东西。

- 这个模型有点简单，特征提取、数据增强等等避免小型图像数据集过拟合的方式都没什么用武之地。

- 很奇怪的一件事是，在小规模样本上使用VGG16对数据增强后的图片进行训练的效果并不好，只有0.8左右的准确率。一开始以为是CPU进行浮点计算的精度不够，但换成GPU之后还是这个情况。为什么呢？

- TensorFlow是基于图来构建计算的，如果不手动释放计算图，会导致占用的内存越来越多，计算速度越来越慢；这是本次实验中踩的最大的坑。Keras对此的封装非常到位：

  ```python
  sess = tf.Session()
  K.set_session(sess)
  foo = K.get_session().run(foo)
  K.clear_session()
  ```

#### 致谢

感谢赵文祺同学给我推荐了深度学习的入门书籍，在我尝试使用预训练网络时借给我GPU，并且帮助我优化了算法的速度。

感谢助教纠正了我的一些愚蠢的错误。

在此表示诚挚的感谢。

### 参考资料

[1] Chollet F. Deep Learning mit Python und Keras: Das Praxis-Handbuch vom Entwickler der Keras-Bibliothek[M]. MITP-Verlags GmbH & Co. KG, 2018. 

[2] Brownlee J. Deep learning with python: Develop deep learning models on theano and tensorflow using keras[M]. Machine Learning Mastery, 2016. 

[3] Srivastava N, Hinton G, Krizhevsky A, et al. Dropout: a simple way to prevent neural networks from overfitting[J]. The journal of machine learning research, 2014, 15(1): 1929-1958. 

[4] Kingma D P, Ba J. Adam: A method for stochastic optimization[J]. arXiv preprint arXiv:1412.6980, 2014. 

[5] Tieleman T, Hinton G. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude[J]. COURSERA: Neural networks for machine learning, 2012, 4(2): 26-31. 

[6] Wang Z, Bovik A C, Sheikh H R, et al. Image quality assessment: from error visibility to structural similarity[J]. IEEE transactions on image processing, 2004, 13(4): 600-612. 

[7] Xiao C, Li B, Zhu J Y, et al. Generating adversarial examples with adversarial networks[J]. arXiv preprint arXiv:1801.02610, 2018. 

[8] Goodfellow I J, Shlens J, Szegedy C. Explaining and harnessing adversarial examples[J]. arXiv preprint arXiv:1412.6572, 2014. 

[9] Bottou L. Large-scale machine learning with stochastic gradient descent[M]//Proceedings of COMPSTAT'2010. Physica-Verlag HD, 2010: 177-186. 

[10] Dong Y, Liao F, Pang T, et al. Boosting adversarial attacks with momentum[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 9185-9193. 

[11] Moosavi-Dezfooli S M, Fawzi A, Frossard P. Deepfool: a simple and accurate method to fool deep neural networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 2574-2582. 

[12] Su J, Vargas D V, Sakurai K. One pixel attack for fooling deep neural networks[J]. IEEE Transactions on Evolutionary Computation, 2019. [12] Su J, Vargas D V, Sakurai K. One pixel attack for fooling deep neural networks[J]. IEEE Transactions on Evolutionary Computation, 2019. 