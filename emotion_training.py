import pandas as pd
import numpy as np
import pickle
import json

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

import os

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(4)

# print if GPU is available
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

# set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# model settings
overwrite = False # will overwrite the current model
model_name = "model-2"
model_folder = f"./emotion_models/{model_name}/"

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

# loading the data
emotion_dataset_full = pd.read_csv("datasets/emotion_cleaned.csv")

"""
Since I've spent 8 weeks (!) trying to get my GPU running, bought 3 new GPUs (RTX 3080 Ti, RX 6950 XT, and RX 7900 XT) and none of them would work with TensorFlow, I'll have to use my CPU for training. Since a single Epoch with the full dataset will take over 20 Minutes (on a Ryzen 7950X), I'll be using a subset of 5% or 10% of the entire dataset.
"""
subset_size = 1
emotion_dataset = emotion_dataset_full.sample(frac=subset_size, random_state=42).reset_index(drop=True)

# loading the cleaned dataset
text_data = emotion_dataset["text"]
label_data = emotion_dataset["emotion"]

# Tokenize all words in text_data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)

# Store tokenizer
with open(os.path.join(model_folder, f"tokenizer-{model_name}.pickle"), "wb") as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

# Tokenize the texts and add padding
text_sequences = tokenizer.texts_to_sequences(text_data)
max_len = max([len(x) for x in text_sequences])
padded_sequences = pad_sequences(text_sequences, maxlen=max_len, padding="post")

# Encoding the labels with LabelEncoder
label_encoder = LabelBinarizer()
train_label_encoded = label_encoder.fit_transform(label_data)

# Store label encoder
with open(os.path.join(model_folder, f"label_encoder-{model_name}.pickle"), "wb") as label_encoder_file:
    pickle.dump(label_encoder, label_encoder_file)

# Model settings
epochs = 16
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 512
input_len = max_len
lstm_units = 256
dropout = 0.3
end_nodes = train_label_encoded.shape[1]
activation = "softmax"
loss = "categorical_crossentropy"
metrics = ["accuracy"]
batch_size = 32
validation_split = 0.1
learning_rate = 0.01

optimizer = Adam(learning_rate=learning_rate)

# Building or loading the model
if os.path.isfile(os.path.join(model_folder, f"{model_name}.h5")) and not overwrite:
    model = tf.keras.models.load_model(os.path.join(model_folder, f"{model_name}.h5"))
    print(f"Loaded model {model_name}.h5")
else:
    with open(os.path.join(model_folder, f"config-{model_name}.json"), "w") as config_json:
        json_dict = {
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "input_len": input_len,
            "lstm_units": lstm_units,
            "dropout": dropout,
            "end_nodes": end_nodes,
            "activation": activation,
            "loss": loss,
            "metrics": metrics,
            "batch_size": batch_size,
            "validation_split": validation_split,
            "learning_rate": learning_rate
        }
        json.dump(json_dict, config_json, indent=4)

    model = tf.keras.models.Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        Bidirectional(LSTM(units=lstm_units, return_sequences=True)),
        Bidirectional(LSTM(units=lstm_units)),
        Dropout(dropout),
        Dense(256, activation='relu'),
        Dropout(dropout),
        Dense(end_nodes, activation=activation)
    ])

optimizer = Adam(learning_rate=0.005)
early_stopping = EarlyStopping(patience=10, monitor="val_loss", restore_best_weights=True)

# Compile the model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(padded_sequences, train_label_encoded, validation_split=validation_split, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])

model.save(os.path.join(model_folder, f"{model_name}.h5"))