import pandas as pd
import numpy as np

emotion_df = pd.read_csv('datasets/go_emotions_dataset.csv')

print(emotion_df.shape)

# dropping all missing values before further processing
emotion_df.dropna(inplace=True)

# removing all columns where "example_very_unclear" is true
emotion_df = emotion_df[emotion_df["example_very_unclear"] == False]

# dropping the "id" and "example_very_unclear" columns, as they are no longer needed for analysis
emotion_df.drop(columns= ["id", "example_very_unclear"], inplace=True)

# listing all emotion columns
emotion_columns =['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 
                  'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
                  'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
                  'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

# calculating the sum of emotions per row
emotion_sums = emotion_df[emotion_columns].sum(axis=1)

# dropping/deleting all rows with 2+ or 0 emotions
emotion_df.drop(emotion_df[emotion_sums >= 2].index, inplace=True)
emotion_df.drop(emotion_df[emotion_sums == 0].index, inplace=True)

# removing all spaces
emotion_df["text"] = emotion_df["text"].apply(lambda x: x.strip())

# filtering out the emotions
emotions_mask = emotion_df[emotion_columns] == 1
emotion_labels = np.where(emotions_mask, emotion_columns, "")
mask = emotion_labels != ""
emotion_labels = emotion_labels[mask]

# create new df with only text and emotion labels
emotion_df = pd.DataFrame({"text": emotion_df["text"], "emotion": emotion_labels})

# storing the cleaned dataset
emotion_df.to_csv("datasets/emotion_cleaned.csv", index=False)

print(emotion_df.shape)