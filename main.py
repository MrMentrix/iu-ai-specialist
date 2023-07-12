import pickle
import json
import numpy as np

from sklearn.preprocessing import LabelEncoder

from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the emotion model and tokenizer
emotion_model_version = "model-3"
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

while True:

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

    top_3_indices = np.argsort(emotion_prediction[0])[:-4:-1]
    top_3_emotions = emotion_label_encoder.inverse_transform(top_3_indices)
    top_3_probabilities = emotion_prediction[0][top_3_indices]

    for emotion, probability in zip(top_3_emotions, top_3_probabilities):
        print(f"{emotion[0].upper()}{emotion[1:]}: Probability {round(probability*100, 2)}%")

    # Output predictions
    print("Sarcasm Prediction:", "Sarcastic" if sarcasm_prediction[0][0] >= 0.5 else "Not Sarcastic", f"(Probability: {sarcasm_prediction[0][0] * 100:.2f}%)")
