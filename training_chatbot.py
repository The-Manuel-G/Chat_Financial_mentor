import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import random

# Carga y procesamiento de datos
with open("intents_spanish.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

words = set()
classes = set()
documents = []
ignore_words = {"?", "!"}

# Procesamiento eficiente de datos
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokeniza y procesa cada patrón
        tokens = nltk.word_tokenize(pattern)
        lemmatized = [lemmatizer.lemmatize(token.lower()) for token in tokens if token not in ignore_words]
        words.update(lemmatized)
        documents.append((lemmatized, intent['tag']))
        classes.add(intent['tag'])

# Preparación final de datos
words = sorted(words)
classes = sorted(classes)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Creación de conjuntos de entrenamiento
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = [1 if word in doc[0] else 0 for word in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
train_x, train_y = zip(*training)
train_x, train_y = np.array(train_x), np.array(train_y)

# Configuración y entrenamiento del modelo
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

lr_schedule = ExponentialDecay(initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.9)
sgd = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chat_model.h5')

print("Modelo Creado")
