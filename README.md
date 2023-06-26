# iu-ai-specialist
Repository for Artificial Intelligence Project for IU

# Virtual Environment Setup
To successfully replicate this project, it is recommended to set up an isolated virtual environment. The prerequisits include Python/Python3, Python-Venv and Python-Pip.

1. Clone this repository to any place you chose
2. Open a terminal and locate the folder you cloned this repository to
3. Use `python -m venv <path>` in the terminal
4. Activate the environment, Linux: `source venv/bin/activate`; Windows: `venv\scripts\activate` 
5. Run `pip install -r requirements.txt`

# Datasets
Here is a quick overview of all datasets that have been used:

1. [datasets for natural language processing](https://www.kaggle.com/datasets/toygarr/datasets-for-natural-language-processing) was used to build the sarcasm detector
2. [Go Emotions: Google Emotions Dataset](https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset) was used to build the emotion detector

# Building the Machine Learning Models
Testing showed that it would make more sense to perform 2 individual tests, one for the general emotions, and one for sarcasm, which will print the likelihood of a certain text being sarcastic. Since it can be hard to automatically detect subtle sarcasm, instead of a binary classification of, e.g., "sarcasm/no sarcasm", a linear probability of 0%-100% for sarcasm will be returned. The same approach is taken for emotions, where a total of 28 emotions are predicted on a 0%-100% scale. This is, since a review can be, e.g., both happy about a cool feature while sad that the product broke. If the emotional state is very unclear, this will be highlighted.


# Sarcasm Model

## Sarcasm Analysis Dataset
For the sarcasm analysis, the following dataset from Kaggle was used: [datasets for natural language processing](https://www.kaggle.com/datasets/toygarr/datasets-for-natural-language-processing). The data contains multiple folders, one of them including a sarcasm training and testing set.

### Sarcasm Analysis Preparations
First, the data was cleaned in `sarcasm_cleaning.py`, by simply removing all missing values. Since checking for any semantic errors would require tons of manual work, this step was skipped. Both the original `sarcasm_train.csv` and `sarcasm_test.csv` were concatenated into `sarcasm_cleaned.csv`, to enable any test-train-ratio during the validation process. To ensure reproducability, a seed of `42` was chosen for both numpy and tensorflow, as it may give the answer to everything.

To store multiple model architectures for later comparison, all sarcasm models were stored in the `sarcasm-models` folder. The current model architecture configuration can be viewed in `sarcarsm_training.py`.


# Emotion Model

## Emotion Analysis Dataset
For the emotion analysis, another dataset from Kaggle was used: [Go Emotions: Google Emotions Dataset](https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset). This data contains 28 different emotions and a classification whether or not a text is "very unclear".

### Emotion Analysis Preparations

During the preparations, a seed is set for `42`, as a uniform seed is practical. Also, all missing values are removed in `emotion_cleaning.py`, before the final dataset `emotion_cleaned.csv` ist stored in the data folder.

To store all model architectures during the training process, all emotion models were stored in `emotion-models`.

### Emotion Data Cleaning & Preprocessing

Some datapoints had multiple emotions assigned to them, while others only had a single emotion. The clear majority of the dataset had a single emotion assigned, single ones even up to 12 emotions. To ensure a uniformous label, only the datapoints with a single emotion were taken. To reduce complexity, the 28 individual binary emotion columns/features have been reduced to a single "emotion" column. All missing values, etc., have been removed from the dataset.

Distribution of amout of emotions per datapoint:
{1: 175231, 2: 31187, 3: 4218, 4: 399, 7: 20, 6: 53, 5: 106, 9: 3, 8: 6, 10: 1, 12: 1}

Additionally, the `id` column was removed, all rows with `emotion_df["example_very_unclear"] == True` were removed, and the `example_very_unclear` column has also been removed, to free the dataset of unneeded information. This reduced the dataset from a `(211225, 31)` shape to `(171820, 2)`, removing exactly 4,000 datapoints, and 2 columns.

Just like with the sarcasm model, the individual architectures have been stored in the `emotion_models` folder. Since the dataset was a lot larger and a single epoch took about 30 minutes, fewer models have been trained. 


# Testing the Models

By running `test.py`, you'll be able to input single sentences for testing. Examples can be found in `samples.md`.