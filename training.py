import random
import json
import pickle
import numpy as np
import nltk
from keras.src.layers import InputLayer
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Dense, Dropout
from keras._tf_keras.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

nltk.download('wordnet')

lematizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Process intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)  # Tokenize the pattern
        words.extend(word_list)
        documents.append((word_list, intent['tag']))  # Append the pattern with its tag
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and remove duplicates
words = [lematizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))  # Sorted unique words
classes = sorted(set(classes))  # Sorted unique classes

# Save words and classes for later use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Initialize training data
training = []
output_empty = [0] * len(classes)  # Empty one-hot encoded vector

# Create the bag of words and one-hot encoded output vectors
for document in documents:
    bag = [0] * len(words)  # Initialize a zero vector for the bag of words
    word_patterns = [lematizer.lemmatize(word.lower()) for word in document[0]]  # Lemmatize the words in the pattern

    # Populate the bag of words
    for word in word_patterns:
        if word in words:
            index = words.index(word)  # Find the index of the word in the vocabulary
            bag[index] = 1  # Set the corresponding index to 1 in the bag

    output_row = list(output_empty)  # Copy the empty output vector
    output_row[classes.index(document[1])] = 1  # Set the corresponding index for the class to 1

    training.append([bag, output_row])  # Append the bag and the output vector to the training set

# Shuffle the training data
random.shuffle(training)

# Convert training data to numpy arrays
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]))  # Features (bags of words)
train_y = np.array(list(training[:, 1]))  # Labels (one-hot encoded classes)

# Build the model
model = Sequential()
model.add(InputLayer(input_shape=(len(train_x[0]),)))  # Input shape is the length of the feature vectors
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))  # Output layer with softmax activation

# Compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('model.h5', hist)

print('Done')
