# iu-ai-specialist
Repository for Artificial Intelligence Project for IU

# Virtual Environment Setup
To successfully replicate this project, it is recommended to set up an isolated virtual environment. The assumption is made that you are using Windows.

1. Clone this repository to any place you chose
2. Open a terminal and locate the folder you cloned this repository to
3. Use `python -m venv venv` in the terminal
4. Run `pip install -r requirements.txt`

# Datasets
Since I couldn't find a reliable dataset with customer reviews and existing emotion-labels, I decided to first use regular texts to train the model on emotion detection. Then, I would use another dataset with reviews to detect emotions within reviews.

## Training Emotion Detection

1. "[Emotions in text](https://www.kaggle.com/datasets/ishantjuyal/emotions-in-text)"
> This dataset was taken from Kaggle and consists of two columns, Text and Emotions. It is unclear where the data originates from.  
> There are 6 different emotions, which are:
> - sadness
> - anger
> - love
> - surprise
> - fear
> - happy

2. "[Emotion Detection from Text](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text)"
> This datasit was taken from Kaggle and consists of three columns, tweet_id, sentiment, and content. The data originates from Tweets.  
> There are 13 different emotions, which are:
> - empty
> - sadness
> - enthusiasm
> - neutral
> - worry
> - surprise
> - love
> - fun
> - hate
> - happiness
> - boredom
> - relief
> - anger

Some of the emotions overlap between the datasets, which are what can be considered "common emotions", such as happiness, fear, anger, surprise, and love. These emotions will be merged together during the merging process. The goal of this merging of the two datasets is to provide a larger training dataset for the model, which may eventually lead to better performance.

### Merging the datasets
To merge the datasets, the following steps will be applied:
1. Clean both datasets individually (appeared to be not needed, since the datasets were of good quality and had no missing values)
2. Merge both datasets into one "emotion_data.csv" dataset, where the "Text" and "content" as well as the "Emotions" and "sentiment" columns will be appended.

**All these steps can be found in "merging.py"!**

### Accounting for Irony and Sarcasm
Since an additional objective of this task is to account for irony and sarcasm as well, a third training dataset from Kaggle was used:
[Tweets with Sarcasm and Irony](https://www.kaggle.com/datasets/nikhiljohnk/tweets-with-sarcasm-and-irony)

The dataset contains 4 classifications of meaning behind a text, which are "figurative", "irony", "regular", and "sarcasm".

## Testing Dataset

The testing dataset being used is "[amazon_review_full](https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)", which is provided by Xiang Zhang on their Google Drive. I found it via a [reference on Kaggle](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews).

# Training Preparations
For this project, two models will be created. One to detect irony and sarcasm, and a second model to detect the emotions of a given text.
While I first tried using sklearn, I decided to refactor my code and use tensorflow instead. The reason behind this is that due to longer training and testing times, I wanted to utilize my GPU instead of solely running the code on my CPU. However, since the combination of Windows, Nvidia, and Microsoft Visual Studio made it pretty much impossible to set up a working environment for `pycuda`, I decided to give it another shot with tensorflow.

## Irony & Sarcasm Detector
