import pandas as pd

sarcasm_test = pd.read_csv("./datasets/sarcasm_test.csv")
sarcasm_train = pd.read_csv("./datasets/sarcasm_train.csv")

sarcasm_test.dropna(inplace=True)
sarcasm_train.dropna(inplace=True)

dataset = pd.concat([sarcasm_test, sarcasm_train], ignore_index=True)

dataset.rename(columns={"Y": "sarcasm"}, inplace=True)
dataset.replace({"sarcasm": {1: "sarcasm", 0: "honest"}}, inplace=True)

dataset.to_csv("./datasets/sarcasm_cleaned.csv", index=False)