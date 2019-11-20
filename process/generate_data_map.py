from keras.datasets import fashion_mnist
import pickle
import numpy as np
from tqdm import tqdm

(train_data, train_labels), (t_data, test_labels) = fashion_mnist.load_data()
test_data = np.load("../test_data/test_data.npy")

data_label_map = dict()
data_index_map = dict()
for i in tqdm(range(len(test_data))):
    data_label_map[str(test_data[i])] = test_labels[i]
    data_index_map[str(test_data[i])] = i

with open("data_label_map.dat", "wb") as l_map:
    pickle.dump(data_label_map, l_map)
with open("data_index_map.dat", "wb") as l_map:
    pickle.dump(data_index_map, l_map)
