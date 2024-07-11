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

# Inicializar lematizador y cargar intents
lemmatizer = WordNetLemmatizer()
with open("intents_spanish.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

words = set()
classes = set()
documents = []
ignore_words = {"?", "!"}

# Procesamiento eficiente de datos
for intent in intents['intents']:
    # Verificar si la clave 'patterns' existe
    if 'patterns' not in intent or 'tag' not in intent:
        continue  # Saltar si no existe
    
    for pattern in intent['patterns']:
        # Tokenizar y procesar cada patrón
        tokens = nltk.word_tokenize(pattern)
        lematizados = [lemmatizer.lemmatize(token.lower()) for token in tokens if token not in ignore_words]
        words.update(lematizados)
        documents.append((lematizados, intent['tag']))
        
        # Asegurarse de que el tag sea una cadena
        if isinstance(intent['tag'], list):
            for tag in intent['tag']:
                if isinstance(tag, str):
                    classes.add(tag)
        else:
            if isinstance(intent['tag'], str):
                classes.add(intent['tag'])

# Preparación final de datos
words = sorted(words)
classes = sorted(classes)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Crear conjuntos de entrenamiento
training = []
output_empty = [0] * len(classes)

for doc in documents:
    try:
        bag = [1 if word in doc[0] else 0 for word in words]
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])
    except ValueError:
        print(f"Etiqueta no encontrada en clases: {doc[1]}")
        continue

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
