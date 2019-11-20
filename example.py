from keras.models import load_model
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import numpy as np
from generate import generate
from tqdm import tqdm

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
for i in tqdm(range(len(test_data))):
    if attack_success(test_data[i], attack_data[i]):
        success_count += 1
print("Attack Success:", success_count / len(test_data))

# evaluate model
val_loss, val_acc = model.evaluate(attack_data, to_categorical(test_labels), verbose=0)
print("Model Loss after Attack:", val_loss)
print("Model Accuracy after Attack:", val_acc)
