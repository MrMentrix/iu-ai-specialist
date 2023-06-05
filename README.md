# iu-ai-specialist
Repository for Artificial Intelligence Project for IU

# Virtual Environment Setup
To successfully replicate this project, it is recommended to set up an isolated virtual environment. The assumption is made that you are using Windows.

1. Clone this repository to any place you chose
2. Open a terminal and locate the folder you cloned this repository to
3. Use `python -m venv venv` in the terminal
4. Run `pip install -r requirements.txt`

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

To store all data during the training process and iteration, all sarcasm models were stored in the `sarcasm-models` folder.

### Sarcasm Analysis Machine Learning Model
After several iterations, a model with the following architecture was chosen:
First, an embedding layer converts the text data into dense vector representations. Two Bidirectional layers will then process the input and store information in Long Short-Term Memory. A dropout layer then randomly removes inputs (sets them to 0) to avoid overfitting, before the all values are given to a dense layer, which is repeated once.

With this architecture, all relevant information should be captured while overfitting is being avoided.


# Emotion Model

## Emotion Analysis Dataset
For the emotion analysis, another dataset from Kaggle was used: [Go Emotions: Google Emotions Dataset](https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset). This data contains 28 different emotions and a classification whether or not a text is "very unclear".

### Emotion Analysis Preparations

During the preparations, a seed is set for `42`, as a uniform seed is practical. Also, all missing values are removed in `emotion_cleaning.py`, before the final dataset `emotion_cleaned.csv` ist stored in the data folder.

To store all model architectures during the training process, all emotion models were stored in `emotion-models`.

### Emotion Data Cleaning & Preprocessing

To determine whether the model should predict a single emotion, or make predictions on all emotions and give a likelihood, the training data was analyzed to see how many data points would have multiple emotions associated with them. The results are:
{1: 175231, 2: 31187, 3: 4218, 4: 399, 7: 20, 6: 53, 5: 106, 9: 3, 8: 6, 10: 1, 12: 1}

This indicates that while the majority of datapoints has only one emotion associated with them, there are several (a little over 35,000) datapoints where multiple emotions are associated. However, 3 emotions appears to be still likely, while 4 or more emotions seems to be very rare. When looking at some more extreme cases, where there have been 10 or 12 emotions, it becomes clear, that these classifications cannot be fully correct.

Examples:
1. "At least you should have helped out since you had rudely interrupted him."
Emotions: ['admiration', 'anger', 'caring', 'curiosity', 'disgust', 'embarrassment', 'nervousness', 'realization', 'remorse', 'sadness']

2. "Two or three anti depressants before I told them a lie about how I tried my moms valium and it worked"
Emotions: ['admiration', 'approval', 'curiosity', 'disappointment', 'embarrassment', 'fear', 'gratitude', 'nervousness', 'optimism', 'pride', 'realization', 'remorse']

Looking at example 1., personally, I only agree with "caring" based on the emotion classification, making the other 9 emotions obsolete. For example 2., I would classify the sentence as rather "neutral", possibly "gratitude". These two examples lead me to the decision to remove all datapoints with more than 3 emotions.

Additionally, the `id` column was removed, all rows with `emotion_df["example_very_unclear"] == True` were removed, and the `example_very_unclear` column has also been removed, to free the dataset of unneeded information. This reduced the dataset from a `(211225, 31)` shape to `(207225, 29)`, removing exactly 4,000 datapoints, and 2 columns.

