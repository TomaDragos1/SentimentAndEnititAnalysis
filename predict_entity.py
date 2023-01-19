import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.callbacks import ModelCheckpoint


def predict(review):
    # Read the dataset
    data = pd.read_csv("train_dataset.csv", nrows=1000)
    # Clean the special character and make the entity easier
    # (0, 1, 2, 3 instead of text)
    inputs = data['Text'].str.replace('[^a-zA-Z ]', '')
    entity_mapping = {'movie': 0, 'place': 1, 'app': 2, 'product': 3}
    data['Entity'] = data['Entity'].map(entity_mapping)
    labels = data['Entity']

    # Read the dataset
    test_data = pd.read_csv("test_dataset.csv", nrows=200)
    # Clean the special character and make the entity easier
    # (0, 1, 2, 3 instead of text)
    test_inputs = test_data['Text'].str.replace('[^a-zA-Z ]', '')
    entity_mapping = {'movie': 0, 'place': 1, 'app': 2, 'product': 3}
    test_data['Entity'] = test_data['Entity'].map(entity_mapping)
    test_labels = test_data['Entity']

    # Transform them in lists
    training_sentences = []
    training_labels = []
    testing_sentences = []
    testing_labels = []
    for row in inputs:
        training_sentences.append(str(row))
    for row in labels:
        training_labels.append(row)
    for row in test_inputs:
        testing_sentences.append(str(row))
    for row in test_labels:
        testing_labels.append(row)

    # Create a tokenizer with a vocabulary of 40000
    tokenizer = Tokenizer(num_words=40000, oov_token="<OOV>")
    # Fit the tokenizer on the training data
    tokenizer.fit_on_texts(inputs)

    # Create the model and it's layers
    model = Sequential()
    # Add an Embedding layer as the first layer in the model
    # Input dimension is set to 40000 as per the tokenizer
    # Output dimension is set to 16
    # Input length is set to 128, the same as maxlen used in pad_sequences
    model.add(Embedding(input_dim=40000, output_dim=16, input_length=128))
    # Add an LSTM layer with 16 units and dropout rate of 0.5
    model.add(LSTM(units=16, dropout=0.5))
    # Add a Dense layer with 4 units and softmax activation for multi classification
    model.add(Dense(units=4, activation='softmax'))

    # Compile the model with an optimizer and a loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Tokenize the inputs
    sequences = tokenizer.texts_to_sequences(training_sentences)
    input_data = pad_sequences(sequences, maxlen=128, truncating='post')
    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    test_data = pad_sequences(testing_sequences, maxlen=128, truncating='post')

    # One-hot encode the labels
    labels = tf.keras.utils.to_categorical(labels, num_classes=4)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=4)
    # Transform the labels in arrays
    labels = np.array(labels)
    test_labels = np.array(test_labels)

    # Checkpoint that saves the best model
    checkpoint1 = ModelCheckpoint("best_ent_model.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                  mode='max')

    # Start the fitting
    model.fit(input_data, labels, epochs=20, validation_data=(test_data, test_labels),
              callbacks=[checkpoint1], validation_split=0.2, shuffle=True)

    # Tokenize the review given
    review_seq = tokenizer.texts_to_sequences([review])
    review_pad = pad_sequences(review_seq, maxlen=128)

    # Predict the entity
    prediction = model.predict(review_pad)

    return prediction[0]
