import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os
import pickle
import json

# setting seed for LabelEncoder and Tokenizer
np.random.seed(42)
tf.random.set_seed(42)

OVERWRITE = True # overwrite current model version
EPOCHS = 20
MODEL_NAME = "model-5"
MODEL_FOLDER = f"./models/{MODEL_NAME}/"

if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

sarcasm_dataset = pd.read_csv("./datasets/cleaned_sarcasm.csv")

# splitting all data into the text input and the label or "class"
sarcasm_text = sarcasm_dataset["text"]
sarcasm_label = sarcasm_dataset["sarcasm"]

# tokenize all words in sarcasm_text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sarcasm_text)

# store tokenizer
with open(os.path.join(MODEL_FOLDER, f"tokenizer-{MODEL_NAME}.pickle"), "wb") as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

# tokenize the texts and add padding
text_sequences = tokenizer.texts_to_sequences(sarcasm_text)
max_len = max(len(seq) for seq in text_sequences)
padded_text_sequences = pad_sequences(text_sequences, maxlen=max_len, padding="post")

# encoding the labels with LabelEncoder
label_encoder = LabelEncoder()
train_label_encoded = label_encoder.fit_transform(sarcasm_label)

with open(os.path.join(MODEL_FOLDER, f"label_encoder-{MODEL_NAME}.pickle"), "wb") as label_encoder_file:
    pickle.dump(label_encoder, label_encoder_file)

# all model settings
vocab_size = len(tokenizer.word_index) + 1
output_dim = 50
input_length = max_len
LSTMunits = 256
dropout = 0.3
DenseUnits = 1
activation = "sigmoid"
loss = "binary_crossentropy"
metrics = ["accuracy"]
batch_size = 256
validation_split = 0.05
learning_rate = 0.001

optimizer = Adam(learning_rate=learning_rate)

# building or loading the sequential machine learning model
if os.path.isfile(os.path.join(MODEL_FOLDER, f"{MODEL_NAME}.h5")) and OVERWRITE is False:
    model = load_model(os.path.join(MODEL_FOLDER, f"{MODEL_NAME}.h5"))
else:

    with open(os.path.join(MODEL_FOLDER, f"config-{MODEL_NAME}.json"), "w") as config_json:
        json_dict = {"vocab_size": vocab_size, "output_dim": output_dim, "input_length": input_length, "LSTMunits": LSTMunits, "dropout": dropout, "DenseUnits": DenseUnits, "activation": activation, "loss": loss, "metrics": metrics, "batch_size": batch_size, "validation_split": validation_split, "learning_rate": learning_rate}
        json.dump(json_dict, config_json, indent=4)

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=output_dim, input_length=input_length),
        Bidirectional(LSTM(units=LSTMunits, return_sequences=True)),
        Bidirectional(LSTM(units=LSTMunits)),
        Dropout(dropout),
        Dense(units=DenseUnits, activation=activation)
    ])

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# prevent long epoch cycles if no significant improvement is made
early_stopping = EarlyStopping(patience=3, monitor="val_loss", restore_best_weights=True)

history = model.fit(padded_text_sequences, train_label_encoded, epochs=EPOCHS, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping])

model.save(os.path.join(MODEL_FOLDER, f"{MODEL_NAME}.h5"))