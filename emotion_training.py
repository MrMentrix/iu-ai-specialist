import pandas as pd
import numpy as np

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import os

# set seeds for reproducibility
np.random.seed(42)
np.random.seed(42)

# model settings
overwrite = True # will overwrite the current model
epochs = 20
model_name = "model-1"
model_folder = f"./emotion_models/{model_name}/"

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

# loading the data
emotion_dataset = pd.read_csv("datasets/emotion_cleaned.csv")

# loading the cleaned dataset
text_data = emotion_dataset["text"]
label_data = emotion_dataset.iloc[:, 1:]

# Encode the labels
labels = np.array(label_data)

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)

# Convert the text data to sequences
sequences = tokenizer.texts_to_sequences(text_data)

# Pad the sequences to ensure uniform length
max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding="post")

# Define your model
model = tf.keras.models.Sequential([
    # Define your model layers
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model with validation split
model.fit(padded_sequences, labels, validation_split=0.2, epochs=10, batch_size=32)
