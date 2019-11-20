from keras.models import load_model
from keras.datasets import fashion_mnist
import numpy as np
from process.ssim import get_ssim
from generate import generate

# load model
model = load_model("model.hdf5")

# load test data and labels
(train_data, train_labels), (t_data, test_labels) = fashion_mnist.load_data()
test_data = np.load("test_data/test_data.npy")

# generate attack data
attack_data = generate(test_data, (len(test_data), 28, 28, 1))


# judge where attack success
def attack_success(prev, attack):
    prev = prev.reshape((1, prev.shape[0], prev.shape[1], prev.shape[2]))
    attack = attack.reshape((1, attack.shape[0], attack.shape[1], attack.shape[2]))
    prev_pred = np.argmax(model.predict(prev))
    current_pred = np.argmax(model.predict(attack))
    return prev_pred != current_pred


# attack success
success_count = 0
for i in range(len(test_data)):
    if attack_success(test_data[i], attack_data[i]):
        success_count += 1
print("Attack Success:", success_count / len(test_data))

# average SSIM
ssim_sum = 0
for i in range(10000):
    ssim_sum += get_ssim(test_data[i], attack_data[i])
print("Average SSIM:", ssim_sum / 10000)
