import numpy as np
from keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os
import pandas as pd

def predict_sarcasm(inputs, model_name):
    # Check if model folder exists
    if not os.path.exists(f"./sarcasm_models/{model_name}/"):
        return "Model folder not found"

    model_folder = f"./sarcasm_models/{model_name}/"

    # Check if model exists in MODELS_FOLDER
    if not os.path.exists(os.path.join(model_folder, f"{model_name}.h5")):
        return "Model not found"

    # Load the trained model
    model = load_model(os.path.join(model_folder, f"{model_name}.h5"))

    # Load the tokenizer, label encoder, and config
    with open(os.path.join(model_folder, f"tokenizer-{model_name}.pickle"), "rb") as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    with open(os.path.join(model_folder, f"label_encoder-{model_name}.pickle"), "rb") as label_encoder_file:
        label_encoder = pickle.load(label_encoder_file)

    config = json.load(open(os.path.join(model_folder, f"config-{model_name}.json"), "r"))

    # Remove all punctuation using regex, including apostrophes
    inputs = inputs.str.replace(r'[^\w\s]', '').str.lower()

    # Tokenize and pad the input sequences
    input_sequences = tokenizer.texts_to_sequences(inputs)
    padded_sequences = pad_sequences(input_sequences, maxlen=config["input_length"], padding='post')

    # Perform inference and get predictions
    predictions = model.predict(padded_sequences, batch_size=512, verbose=0)

    # create dataframe with predicted label and confidence
    results = pd.DataFrame({
        "confidence": predictions.flatten(),
        "label": label_encoder.inverse_transform((predictions > 0.5).astype(int)).flatten()
    })

    return results