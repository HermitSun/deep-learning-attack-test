from keras.datasets import fashion_mnist
import pickle
import numpy as np
from tqdm import tqdm

(train_data, train_labels), (t_data, test_labels) = fashion_mnist.load_data()
train_data = train_data.astype("float32") / 255
test_data = np.load("../test_data/test_data.npy")

data_label_map = dict()
data_index_map = dict()
for i in tqdm(range(len(train_data))):
    data_label_map[str(train_data[i])] = train_data[i]
    data_index_map[str(train_data[i])] = i
for i in tqdm(range(len(test_data))):
    data_label_map[str(test_data[i])] = test_labels[i]
    data_index_map[str(test_data[i])] = len(train_data) + i

with open("data_label_map.dat", "wb") as l_map:
    pickle.dump(data_label_map, l_map)
with open("data_index_map.dat", "wb") as i_map:
    pickle.dump(data_index_map, i_map)

with open("data_label_map.dat", "rb") as l_map:
    data_label_map = pickle.load(l_map)
with open("data_index_map.dat", "rb") as i_map:
    data_index_map = pickle.load(i_map)
