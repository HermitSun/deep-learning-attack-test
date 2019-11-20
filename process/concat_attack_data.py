from keras.models import load_model
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import numpy as np
from process.ssim import get_ssim
from tqdm import tqdm

model = load_model("../model.hdf5")
(train_data, train_labels), (t_data, test_labels) = fashion_mnist.load_data()
test_data = np.load("../test_data/test_data.npy")

attack_data = np.array([]).reshape((0, 28, 28, 1))
for i in range(1, 9):
    path = "../attack_data/attack_data_m" + str(i) + ".npy"
    attack = np.load(path)
    attack_data = np.concatenate([attack_data, attack])
    print(attack_data.shape)
attack_data = attack_data.astype("float32")

ssim_sum = 0
for i in tqdm(range(10000)):
    ssim_sum += get_ssim(test_data[i], attack_data[i])
print(ssim_sum / 10000)

print(model.evaluate(attack_data, to_categorical(test_labels)))

np.save("../attack_data/attack_data.npy", attack_data)
