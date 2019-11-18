from keras.datasets import fashion_mnist
from keras.models import load_model
from process.ssim import get_ssim
import numpy as np
import time
import pickle

# fix random
seed = 7
np.random.seed(seed)

# load dataset
(train_data, train_labels), (t_data, test_labels) = fashion_mnist.load_data()
test_data = np.load("../test_data/test_data.npy")
# load models
base_model = load_model("../models/base_model.hdf5")
cnn_model = load_model("../models/cnn_model.hdf5")
# load labels
with open("data_label_map.dat", "rb") as l_map:
    data_label_map = pickle.load(l_map)
# 读取断点
attack_data = list(np.load("../attack_data/attack_data.npy"))
has_processed = len(attack_data)
ssim_sum = 0
start = time.time()
for i in range(has_processed + 1, len(test_data)):
    # 需要进行预测，所以转换成(n,28,28,1)
    x = test_data[i].reshape(1, 28, 28, 1)
    # 模型一开始的预测结果
    preds = cnn_model.predict(x)
    initial_class = np.argmax(preds)
    # 对于每个样本，随机抽取150个样本，计算SSIM
    samples = [np.random.randint(0, len(test_data)) for j in range(150)]
    max_ssim = 0
    max_j = 0
    for sample_index in samples:
        # 不是自己并且两者标签不同
        if not i == sample_index and \
                not data_label_map.get(hash(str(test_data[i]))) == data_label_map.get(
                    hash(str(test_data[sample_index]))):
            current_ssim = get_ssim(test_data[i], test_data[sample_index])
            if current_ssim > max_ssim:
                max_ssim = current_ssim
                max_j = sample_index
    # 将SSIM最高的样本作为噪声，合并到原图片上，生成对抗样本
    alpha = 0
    beta = 1 - alpha
    gamma = 0
    attack = test_data[i] * alpha + test_data[max_j] * beta + gamma
    print(get_ssim(test_data[i], attack))

    preds = cnn_model.predict(attack.reshape(1, 28, 28, 1))
    new_class = np.argmax(preds)
    attack_data.append(attack)
    print(initial_class, new_class)

    # 保存，可以用于断点续传
    np.save("../attack_data/attack_data.npy", np.array(attack_data))
    print(i)
# 保存对抗样本
data_attack_map = dict()
for i in range(len(test_data)):
    data_label_map[test_data[i]] = attack_data[i]
with open('data_attack_map', 'wb') as map:
    pickle.dump(data_label_map, map)

end = time.time()
ssim_average = ssim_sum / len(test_data)
attack_data = np.array(attack_data)

print(end - start)
print(ssim_average)
