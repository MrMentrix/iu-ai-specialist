import pickle
import numpy as np
import json

from sklearn.preprocessing import LabelBinarizer

from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the emotion model and tokenizer
emotion_model_version = "model-2"
emotion_model = load_model(f"emotion_models/{emotion_model_version}/{emotion_model_version}.h5")
emotion_tokenizer = pickle.load(open(f"emotion_models/{emotion_model_version}/tokenizer-{emotion_model_version}.pickle", "rb"))
emotion_label_encoder = pickle.load(open(f"emotion_models/{emotion_model_version}/label_encoder-{emotion_model_version}.pickle", "rb"))
emotion_config = json.load(open(f"emotion_models/{emotion_model_version}/config-{emotion_model_version}.json", "r"))
emotion_len = emotion_config["input_len"]

# Load the sarcasm model and tokenizer
sarcasm_model_version = "model-8"
sarcasm_model = load_model(f"sarcasm_models/{sarcasm_model_version}/{sarcasm_model_version}.h5")
sarcasm_tokenizer = pickle.load(open(f"sarcasm_models/{sarcasm_model_version}/tokenizer-{sarcasm_model_version}.pickle", "rb"))
sarcasm_config = json.load(open(f"sarcasm_models/{sarcasm_model_version}/config-{sarcasm_model_version}.json", "r"))
sarcasm_len = sarcasm_config["input_length"]

# User input
user_input = input("Enter a sentence: ")

# Tokenize the input for emotion model
emotion_input = emotion_tokenizer.texts_to_sequences([user_input])
emotion_input = pad_sequences(emotion_input, maxlen=emotion_len, padding="post")

# Tokenize the input for sarcasm model
sarcasm_input = sarcasm_tokenizer.texts_to_sequences([user_input])
sarcasm_input = pad_sequences(sarcasm_input, maxlen=sarcasm_len, padding="post")

# Make predictions
emotion_prediction = emotion_model.predict(emotion_input)
sarcasm_prediction = sarcasm_model.predict(sarcasm_input)

emotion_label_binarizer = LabelBinarizer()
emotion_label_binarizer.fit(emotion_label_encoder.classes_)

# Transform emotion predictions using the label binarizer
emotion_predictions_encoded = emotion_label_binarizer.inverse_transform(emotion_prediction)

# Output predictions
print("Emotion Prediction:", emotion_predictions_encoded)
print("Sarcasm Prediction:", "Sarcastic" if sarcasm_prediction[0][0] >= 0.5 else "Not Sarcastic", f"(Probability: {sarcasm_prediction[0][0] * 100:.2f}%)")
