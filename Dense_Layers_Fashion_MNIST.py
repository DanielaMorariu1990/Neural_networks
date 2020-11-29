""" 
Application of feedforward neural networks using only dense layers.
Predicting fashion items from fashion MNIST data set.

"""

from tensorflow.keras.layers import Dense
from tensorflow.keras import layers

from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import types
import pandas as pd
from tensorflow.keras.models import load_model


fashion = pd.read_csv("./data/fashion-mnist_train.csv")
labels = pd.read_csv("./data/label_names.csv")
fashion.join(labels, on="label")["Description"].value_counts()
fashion.join(labels, on="label")["Description"].value_counts()


# standardize the data
input_data = fashion.iloc[:, 1:]/255
target = fashion[["label"]]


# plottong first 10 observations

plt.figure(figsize=(15, 15))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np.array(input_data.iloc[i, :]).reshape(
        28, 28), cmap=plt.cm.binary)
    plt.xlabel(fashion.join(labels, on="label")["Description"][i])


fashion.head()


X = fashion.iloc[:, 1:]
X.head()


y = fashion.iloc[:, 0]
y.head()


# input shape for sequential neural networks
X.iloc[0].shape


# code target variable (y) to categorical
y_cat = to_categorical(y, num_classes=10)
y_cat.shape


# ### create a dense neural network with 6 layers

m2 = Sequential()


# add layers
m2.add(Dense(units=50, activation='elu', input_shape=(784,)))
for i in range(4):
    m2.add(Dense(units=50, activation='elu'))

m2.add(Dense(units=10, activation="softmax"))


m2.compile(loss=tf.keras.losses.CategoricalCrossentropy(
    from_logits=False), optimizer='adam', metrics=['accuracy'])

m2.summary()


history2 = m2.fit(X, y_cat, batch_size=500, epochs=300, validation_split=0.2)

# save model
m2.save("Dense_layers_300.h5")

# plot the accuracy and the corss-validation accuracy
plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
plt.savefig("Dense_6_layer_Network.png")


m2.evaluate(X, y_cat)
#[0.24980269676952932, 0.95128334]

# ### Add early stopping to the netwok

callback = EarlyStopping(monitor='val_loss', patience=5)
history3 = m2.fit(X, y_cat, batch_size=500, epochs=300,
                  validation_split=0.2, callbacks=[callback])


plt.plot(history3.history['acc'])
plt.plot(history3.history['val_acc'])
plt.savefig("Dense_6_layer_earlyStop.png")

m2.evaluate(X, y_cat)
#[0.23561639170811202, 0.95166665]


# ### Add dropout to the Dense layers
m3 = load_model("Model3_dense_new_calib_dropout.h5")
m3 = Sequential()

m3.add(Dense(units=50, activation='elu', input_shape=(784,)))
m3.add(Dropout(0.2))

for i in range(4):
    m3.add(Dense(units=50, activation='elu'))
    m3.add(Dropout(0.2))

m3.add(Dense(units=10, activation="softmax"))


m3.compile(loss=tf.keras.losses.CategoricalCrossentropy(
    from_logits=False), optimizer='adam', metrics=['accuracy'])

m3.summary()

history4 = m3.fit(X, y_cat, batch_size=500, epochs=250,
                  validation_split=0.2)

m3.save("Model3_dense_new_calib_dropout.h5")
plt.plot(history4.history['acc'])
plt.plot(history4.history['val_acc'])

m3.evaluate(X, y_cat)
#[0.47374578246275584, 0.83168334]

# add dropout layers to the previously calibrated network
m2 = load_model('./saved_models/Dense_layers_300.h5')
m2.add(Dense(units=50, activation='elu', input_shape=(784,)))
m2.add(Dropout(0.2))

for i in range(4):
    m2.add(Dense(units=50, activation='elu'))
    m2.add(Dropout(0.2))

m2.add(Dense(units=10, activation="softmax"))


m2.compile(loss=tf.keras.losses.CategoricalCrossentropy(
    from_logits=False), optimizer='adam', metrics=['accuracy'])

m2.summary()

history5 = m2.fit(X, y_cat, batch_size=500, epochs=250,
                  validation_split=0.2)

plt.plot(history5.history['accuracy'])
plt.plot(history5.history['val_accuracy'])
plt.savefig("dense_with_dropout.png")
m2.save("dropout_layers_dense.h5")

m2.evaluate(X, y_cat)
#[0.2085593044757843, 0.9456999897956848]


# make an early stopping
callback = EarlyStopping(monitor='val_loss', patience=5)
history6 = m2.fit(X, y_cat, batch_size=500, epochs=300,
                  validation_split=0.2, callbacks=[callback])


plt.plot(history6.history['accuracy'])
plt.plot(history6.history['val_accuracy'])

m2.evaluate(X, y_cat)
#[0.2474955916404724, 0.9315166473388672]

# ### Add Image augumentation
fashion.iloc[1:, 0]
image = np.array(fashion.iloc[1, 1:]).reshape(28, 28, 1)

# plot original data
plt.imshow(image, cmap=plt.cm.binary)
image = tf.expand_dims(image, 0)

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
])

# augument original data and reshape original data
for i in range(9):
    augmented_image = data_augmentation(image)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_image[0])
    plt.axis("off")
plt.savefig("Augumented_data.png")


IMG_SIZE = 180
image = np.array(fashion.iloc[1, 1:]).reshape(28, 28, 1)

resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
    # layers.experimental.preprocessing.Rescaling(1./255)
])

# plot comparison
result = resize_and_rescale(image)
_ = plt.imshow(result)
plt.imshow(image)
