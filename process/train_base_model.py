from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import callbacks
import numpy as np


# create base model
# Acc: 0.8852
def create_base_model():
    model = models.Sequential()
    model.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(optimizer="rmsprop",
                  loss="categorical_crossentropy",
                  metrics=["acc"])
    return model


# fix random seed
seed = 7
np.random.seed(seed)
# load dataset
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
# standardize
train_data = train_data.reshape(len(train_data), 28 * 28).astype("float32") / 255
test_data = test_data.reshape(len(test_data), 28 * 28).astype("float32") / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# create network
network = create_base_model()
callbacks_list = [
    callbacks.EarlyStopping(monitor="acc", patience=1),
    callbacks.ModelCheckpoint(
        filepath="../models/base_model.hdf5",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

history = network.fit(
    train_data,
    train_labels,
    validation_split=0.2,
    batch_size=128,
    epochs=10,
    callbacks=callbacks_list,
    verbose=0
)

val_loss, val_acc = network.evaluate(test_data, test_labels, verbose=0)
print("Test Accuracy:", val_acc)
