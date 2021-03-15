import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD

doc = pd.read_csv('Book1.csv')

print(doc.head(10))
# print(doc.info())
# print(doc["x"])
# print(np.array(doc))

doc = np.array(doc)
# print(doc)
# print(doc[:, 2])

# train_x = doc[:, 1:4]
# train_y = doc[:, 4:]


train_x = np.array(doc[:, 1:4])
train_y = np.array(doc[:, 4:])

# train_x = train_x / (np.max(train_x) + 1)
# train_y = train_y / (np.max(train_y) + 1)

print(train_x)
print(train_y)

# inputs = keras.Input(shape=(len(train_x[0]),), name="digits")
inputs = keras.Input(shape=(3,), name="digits")
x = layers.Dense(100, activation="relu", name="dense_1")(inputs)
# x = layers.Dense(3, activation="relu", name="dense_2")(x)
# outputs = layers.Dense(len(train_y[0]), activation="sigmoid", name="predictions")(x)
outputs = layers.Dense(1, name="predictions")(x)
# outputs = layers.Dense(len(train_y[0]), activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

sgd = SGD(lr=0.01, decay=1e-1, momentum=0.9, nesterov=True)
print('******************************************************************* start')
print(model.weights)
print('******************************************************************* end')

model.compile(
    # optimizer='sgd',
    # optimizer=sgd,
    optimizer='adam',
    # loss=keras.losses.SparseCategoricalCrossentropy(),
    # loss='categorical_crossentropy',
    loss='mean_squared_error',
    metrics=['accuracy']
    # metrics=[keras.metrics.Accuracy()]
)

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# hist = model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=5, verbose=1)

hist = model.fit(train_x, train_y, epochs=200, batch_size=50, verbose=1)
model.save('testModel1.h5', hist)

print('****************** after Compile ************************************************* start')
print(model.weights)
print('****************** after Compile ************************************************* end')

# x = np.array([2, 1, 5])
# x = np.array([4, 4, 5])
x = np.array([[4, 4, 5], [2, 1, 5], [12000, 36000000, 5]])
pred = model.predict(x.reshape(3, 3), batch_size=50)
print(pred)

# print('predicted')
# print(pred)
# print('trained')
# print(train_y)
