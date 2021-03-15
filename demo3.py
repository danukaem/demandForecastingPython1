import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

# model = keras.Sequential(
#     [
#         layers.Dense(2, activation="relu", name="layer1"),
#         layers.Dense(3, activation="relu", name="layer2"),
#         layers.Dense(4, name="layer3"),
#     ]
# )

# print(model.layers)
# print(model.layers[0])
# Call model on a test input
# x = tf.ones((3, 3))
# x = np.array([[10,10,10],[10,10,10], [10,10,10]])
# x = np.array([[1, 2, 3],[4, 5, 6], [7, 8, 9]])
# y = model(x)
# print(y)


# model = keras.Sequential()
# lyr=layers.Dense(3, activation='relu', name='first_layer')
# print(lyr.weights)

# model = keras.Sequential()
# model.add(layers.Dense(3, activation='relu', name='first_layer'))
# x = tf.ones((3, 3))
# model(x)
# print("aaaaaaaaaaaaaaa")
# print(model.layers[0].weights)
# print("aaaaaaaaaaaaaaa")

cs = pd.read_csv('chat.csv')
print(cs)
print(cs[1:10][2:5])
# model = keras.Sequential()
# model.add(layers.Dense(3, activation='relu', name='first_layer'))
# x = tf.ones((3, 3))
# model(x)
# print("aaaaaaaaaaaaaaa")
# print(model.layers[0].weights)
# print("aaaaaaaaaaaaaaa")
