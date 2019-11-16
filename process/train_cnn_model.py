from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import callbacks
import numpy as np


# create base model
# Acc: 0.9173
def create_cnn_model():
    # create model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation="relu", input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(28 * 28, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    return model


# fix random seed
seed = 7
np.random.seed(seed)
# load dataset
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
# standardize
train_data = train_data.reshape(len(train_data), 28, 28, 1).astype("float32") / 255
test_data = test_data.reshape(len(test_data), 28, 28, 1).astype("float32") / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# create network
network = create_cnn_model()
callbacks_list = [
    callbacks.EarlyStopping(monitor="acc", patience=1),
    callbacks.ModelCheckpoint(
        filepath="../models/cnn_model.hdf5",
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
    epochs=40,
    callbacks=callbacks_list,
    verbose=0
)

val_loss, val_acc = network.evaluate(test_data, test_labels, verbose=0)
print("Test Accuracy:", val_acc)
