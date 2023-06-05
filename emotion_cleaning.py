import pandas as pd

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

# identifying the rows with 4+ emotions, that need to be removed
rows_to_delete = emotion_df[emotion_sums >= 4].index

# dropping/deleting all rows with 4 or more emotions
emotion_df.drop(rows_to_delete, inplace=True)

# removing all spaces
emotion_df["text"] = emotion_df["text"].apply(lambda x: x.strip())

# storing the cleaned dataset
emotion_df.to_csv("datasets/emotion_cleaned.csv", index=False)

print(emotion_df.shape)