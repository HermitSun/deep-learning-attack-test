from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np

# load test data
(train_data, train_labels), (t_data, test_labels) = fashion_mnist.load_data()
test_data = np.load("../test_data/test_data.npy")
test_labels = to_categorical(test_labels)

sample_num = test_data.shape[0]
width = test_data.shape[1]
height = test_data.shape[2]
channels = test_data.shape[3]

# test base model
model = load_model("../models/base_model.hdf5")
test_data = test_data.reshape(sample_num, width * height)
base_model_val_loss, base_model_val_acc = model.evaluate(test_data, test_labels)
print("Base Model Accuracy:", base_model_val_acc)

# test CNN model
model = load_model("../models/cnn_model.hdf5")
test_data = test_data.reshape(sample_num, width, height, channels)
cnn_model_val_loss, cnn_model_val_acc = model.evaluate(test_data, test_labels)
print("CNN Model Accuracy:", cnn_model_val_acc)

# test final model
model = load_model("../model.hdf5")
test_data = test_data.reshape(sample_num, width, height, channels)
val_loss, val_acc = model.evaluate(test_data, test_labels)
print("Final Model Accuracy:", val_acc)
