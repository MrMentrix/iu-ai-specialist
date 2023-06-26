import pandas as pd
import numpy as np
import pickle
import json

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical  # Added import

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
overwrite = False  # will overwrite the current model
model_name = "model-3"
model_folder = f"./emotion_models/{model_name}/"

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

# loading the data
emotion_dataset_full = pd.read_csv("datasets/emotion_cleaned.csv")

# Subset option
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
label_encoder = LabelEncoder()
train_label_encoded = label_encoder.fit_transform(label_data)

# Convert target data to one-hot encoded format
num_classes = label_encoder.classes_.shape[0]
train_label_encoded = to_categorical(train_label_encoded, num_classes=num_classes)  # Added conversion

# Store label encoder
with open(os.path.join(model_folder, f"label_encoder-{model_name}.pickle"), "wb") as label_encoder_file:
    pickle.dump(label_encoder, label_encoder_file)

# Model settings
epochs = 8
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 256
input_len = max_len
lstm_units = 128
dropout = 0.2
end_nodes = num_classes  # Replaced `label_data.unique().shape[0]` with `num_classes`
activation = "softmax"
loss = "categorical_crossentropy"
metrics = ["accuracy"]
batch_size = 128
validation_split = 0.1
learning_rate = 0.001

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

history = model.fit(padded_sequences, train_label_encoded, validation_split=validation_split,
                    epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])

model.save(os.path.join(model_folder, f"{model_name}.h5"))
