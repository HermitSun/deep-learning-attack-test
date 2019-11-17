from keras.datasets import fashion_mnist
from keras.models import load_model
from process import noises
from process.ssim import get_ssim
import numpy as np
import time

# fix random
seed = 7
np.random.seed(seed)

# load dataset
(train_data, train_labels), (t_data, test_labels) = fashion_mnist.load_data()
test_data = np.load("../test_data/test_data.npy")
# load models
base_model = load_model("../models/base_model.hdf5")
cnn_model = load_model("../models/cnn_model.hdf5")

# generate attack data and calculate time cost
attack_data = []
ssim_sum = 0
start = time.time()
for image in test_data:
    sp_test_image = noises.sp_noise(image, 0.05)
    ssim_sum += get_ssim(sp_test_image, image)
    attack_data.append(sp_test_image)
end = time.time()
ssim_average = ssim_sum / len(test_data)

print(end - start)
print(ssim_average)

result = base_model.predict(attack_data)
print(result)
