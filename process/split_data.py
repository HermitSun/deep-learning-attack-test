from keras.datasets import fashion_mnist
import numpy as np

# load dataset
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
# reshape
test_data = test_data.reshape(len(test_data), 28, 28, 1)
# standardize
test_data = test_data.astype("float32") / 255
# save
np.save("../test_data/test_data.npy", test_data)
