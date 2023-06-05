import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import os
import pickle
import json

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

overwrite = True  # Overwrite current model version
epochs = 20
model_name = "model-7"
model_folder = f"./sarcasm_models/{model_name}/"

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

sarcasm_dataset = pd.read_csv("./datasets/sarcasm_cleaned.csv")

# Splitting all data into the text input and the label or "class"
sarcasm_text = sarcasm_dataset["text"]
sarcasm_label = sarcasm_dataset["sarcasm"]

# Tokenize all words in sarcasm_text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sarcasm_text)

# Store tokenizer
with open(os.path.join(model_folder, f"tokenizer-{model_name}.pickle"), "wb") as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

# Tokenize the texts and add padding
text_sequences = tokenizer.texts_to_sequences(sarcasm_text)
max_len = max(len(seq) for seq in text_sequences)
padded_text_sequences = pad_sequences(text_sequences, maxlen=max_len, padding="post")

# Encoding the labels with LabelEncoder
label_encoder = LabelEncoder()
train_label_encoded = label_encoder.fit_transform(sarcasm_label)

with open(os.path.join(model_folder, f"label_encoder-{model_name}.pickle"), "wb") as label_encoder_file:
    pickle.dump(label_encoder, label_encoder_file)

# All model settings
vocab_size = len(tokenizer.word_index) + 1
output_dim = 100
input_length = max_len
lstm_units = 128
dropout = 0.3
dense_units = 1
activation = "sigmoid"
loss = "binary_crossentropy"
metrics = ["accuracy"]
batch_size = 512
validation_split = 0.05
learning_rate = 0.001

optimizer = Adam(learning_rate=learning_rate)

# Building or loading the sequential machine learning model
if os.path.isfile(os.path.join(model_folder, f"{model_name}.h5")) and not overwrite:
    model = tf.keras.models.load_model(os.path.join(model_folder, f"{model_name}.h5"))
else:
    with open(os.path.join(model_folder, f"config-{model_name}.json"), "w") as config_json:
        json_dict = {
            "vocab_size": vocab_size,
            "output_dim": output_dim,
            "input_length": input_length,
            "lstm_units": lstm_units,
            "dropout": dropout,
            "dense_units": dense_units,
            "activation": activation,
            "loss": loss,
            "metrics": metrics,
            "batch_size": batch_size,
            "validation_split": validation_split,
            "learning_rate": learning_rate
        }
        json.dump(json_dict, config_json, indent=4)

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=output_dim, input_length=input_length),
        Bidirectional(LSTM(units=lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        Bidirectional(LSTM(units=lstm_units, dropout=0.2, recurrent_dropout=0.2)),
        Dropout(dropout),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(units=dense_units, activation=activation)
    ])

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Prevent long epoch cycles if no significant improvement is made
early_stopping = EarlyStopping(patience=3, monitor="val_loss", restore_best_weights=True)

history = model.fit(padded_text_sequences, train_label_encoded, epochs=epochs,
                    batch_size=batch_size, validation_split=validation_split,
                    callbacks=[early_stopping])

model.save(os.path.join(model_folder, f"{model_name}.h5"))
