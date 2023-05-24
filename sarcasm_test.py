import numpy as np
from keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os

while True:
    model_selection = input("Select a model: ")
    if not os.path.isdir(f"./models/{model_selection}"):
        print("invalid model")
    else:
        break

MODEL_NAME = "model-2"
MODEL_FOLDER = f"./models/{MODEL_NAME}/"

# Load the trained model
model = load_model(os.path.join(MODEL_FOLDER, f"{MODEL_NAME}.h5"))

# Load the tokenizer, label encoder, and config
with open(os.path.join(MODEL_FOLDER, f"tokenizer-{MODEL_NAME}.pickle"), "rb") as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

with open(os.path.join(MODEL_FOLDER, f"label_encoder-{MODEL_NAME}.pickle"), "rb") as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

config = json.load(open(os.path.join(MODEL_FOLDER, f"config-{MODEL_NAME}.json"), "r"))

inputs = [
    ("Can't wait to spend my weekend doing laundry and cleaning. So exciting!", "sarcasm"),
    ("I absolutely love waking up early on weekends!", "honest"),
    ("Oh, another meeting? That's exactly what I wanted.", "sarcasm"),
    ("Spending hours in traffic is my favorite hobby. So relaxing!", "sarcasm"),
    ("I won the lottery! Just kidding, it's Monday again.", "sarcasm"),
    ("The weather is perfect for a picnic today. I'm enjoying every moment!", "honest"),
    ("I adore waiting in long lines at the supermarket. It's the highlight of my day!", "sarcasm"),
    ("Getting stuck in rush hour traffic is always a pleasure.", "sarcasm"),
    ("Finally, a day off to relax and do nothing. I'm thrilled!", "honest"),
    ("Wow, another email to respond to? My day is complete!", "sarcasm"),
    ("I'm so grateful for the opportunity to work late on a Friday evening.", "sarcasm"),
    ("Having a good night's sleep is overrated. I prefer insomnia.", "sarcasm"),
    ("I just finished my workout, feeling fantastic and energized!", "honest"),
    ("Another rainy day, just what I needed to make my day perfect!", "sarcasm"),
    ("I'm thrilled to be stuck in the airport for hours. It's a dream come true!", "sarcasm"),
    ("Spending time with my extended family is always so relaxing and stress-free.", "honest"),
    ("What a surprise! My favorite show got canceled. I'm overjoyed!", "sarcasm"),
    ("I had the best time waiting in line at the amusement park. Worth every minute!", "sarcasm"),
    ("I'm so lucky to have such a fantastic boss. Their feedback always motivates me.", "honest"),
    ("My computer crashed right before the deadline. Perfect timing!", "sarcasm"),
    ("I'm excited to start my day with a long and boring meeting.", "sarcasm"),
    ("I love waking up early and going for a run. It's the best way to start the day!", "honest"),
    ("I'm thrilled to spend my vacation at home. So many exciting things to do!", "sarcasm"),
    ("Having a good night's sleep is the key to a productive day. Rested and ready!", "honest"),
    ("I'm so grateful for the opportunity to work overtime on a holiday. It's a dream come true!", "sarcasm"),
    ("Nothing beats the joy of being stuck in traffic on a Friday evening. Pure bliss!", "sarcasm"),
    ("I just received great news! Can't contain my excitement!", "honest"),
    ("Another day of rain. I can't wait to get soaked on my way to work!", "sarcasm"),
    ("I absolutely adore cleaning the house. It's my favorite way to relax!", "sarcasm"),
    ("I have the best luck! Forgot my phone at home today.", "sarcasm"),
    ("I'm looking forward to spending hours on hold with customer service. What fun!", "sarcasm"),
    ("I enjoy eating plain rice for dinner every night. It's a culinary delight!", "sarcasm"),
    ("I'm so lucky to have such supportive colleagues. They make work a joy!", "honest"),
    ("I can't wait to spend my weekend hiking and exploring nature.", "honest")
]

correct = 0
total = len(inputs)

for input in inputs:

    # Tokenize user input
    input_sequence = tokenizer.texts_to_sequences([input][0])
    padded_sequence = pad_sequences(input_sequence, maxlen=config["input_length"], padding='post')

    # Perform inference and get predictions
    predictions = model.predict(padded_sequence)

    try:
        predicted_label_index = np.argmax(predictions, axis=1)[0]
        predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
        confidence = predictions[0][predicted_label_index]
        if confidence >= 0.5:
            prediction = "sarcasm"
        else:
            prediction = "honest"
            confidence = 1 - confidence
    except ValueError as e:
        missing_label = str(e).split(": ")[1]
        print(f"Error: Unseen label encountered: {missing_label}")
        exit()

    if prediction == input[1]:
        correct += 1

print(f"Classified {correct} out of {total} ({round((correct/total)*100, 2)}%)")