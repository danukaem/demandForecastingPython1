import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

from tensorflow import keras
from tensorflow.keras import layers

lemmatizer = WordNetLemmatizer()

porter = PorterStemmer()

intents = json.loads(open("intentCombined.json").read())
# intents = json.loads(open("intent.json").read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [porter.stem(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [porter.stem(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

x = np.array([(1, 5, 66), (8, 23, 6), (8, 63, 552), (4, 558, 31)], dtype='int32')
y = np.array([(1, 5), (6, 2), (4, 46), (48, 51)], dtype='int32')

# train_x = list(training[:, 0])
# train_y = list(training[:, 1])

train_x = list(x)
train_y = list(y)

inputs = keras.Input(shape=(len(train_x[0]),), name="digits")
x = layers.Dense(10, activation="relu", name="dense_1")(inputs)
x = layers.Dense(10, activation="relu", name="dense_2")(x)
outputs = layers.Dense(len(train_y[0]), activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=2000, verbose=1)

model.save('chatbotmodel.h5', hist)
print("Done")
