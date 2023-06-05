from sarcasm_test import predict_sarcasm
import pandas as pd

sarcasm_dataset = pd.read_csv("./datasets/cleaned_sarcasm.csv")

# Take 500 random samples from the dataset
sarcasm_dataset = sarcasm_dataset.sample(n=500).reset_index(drop=True)

# Make predictions on all 500 samples and obtain the results DataFrame
sarcasm_results = predict_sarcasm(sarcasm_dataset["text"], "model-6")
sarcasm_results["original_label"] = sarcasm_dataset["sarcasm"]

print(sarcasm_results)

# Calculate the accuracy of the model
accuracy = (sarcasm_results["label"] == sarcasm_results["original_label"]).sum() / len(sarcasm_results)
print(f"Accuracy: {accuracy}")