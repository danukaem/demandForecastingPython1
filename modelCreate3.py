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

print('intents--------------start')
print(intents)
print('intents--------------end')
print('intents length :  ', len(intents))

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

print('words--------------start')
print(words)
print('words--------------end')
print('words length :  ', len(words))

print('documents--------------start')
print(documents)
print('documents--------------end')
print('documents length :  ', len(documents))

words = [porter.stem(word) for word in words if word not in ignore_letters]

words = sorted(set(words))

print('words--------------start')
print(words)
print('words--------------end')
print('words length :  ', len(words))

classes = sorted(set(classes))

print('classes--------------start')
print(classes)
print('classes--------------end')
print('classes length:  ', len(classes))

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

print('training--------------start')
print(training)
print('training--------------end')
print('training length :  ', len(training))

train_x = list(training[:, 0])
train_y = list(training[:, 1])

print('train_x--------------start')
print(train_x)
print('train_x--------------end')
print('train_x length :  ', len(train_x))

print('train_y--------------start')
print(train_y)
print('train_y--------------end')
print('train_y length :  ', len(train_y))

print('len(train_x[0]) length :  ', len(train_x[0]))
print('len(train_y[0]) length :  ', len(train_y[0]))

inputs = keras.Input(shape=(len(train_x[0]),), name="digits")
x = layers.Dense(10000, activation="relu", name="dense_1")(inputs)
x = layers.Dense(1000, activation="relu", name="dense_2")(x)
outputs = layers.Dense(len(train_y[0]), activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# model = Sequential()
# model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(train_y[0]), activation='softmax'))

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

hist = model.fit(np.array(train_x), np.array(train_y), epochs=10, batch_size=100, verbose=1)

print('hist--------------start')
print(hist)
print('hist--------------end')

model.save('chatbotmodel.h5', hist)
print("Done")
