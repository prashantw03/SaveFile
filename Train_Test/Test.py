import time

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import os,datetime

os.makedirs('../data', exist_ok=True)

class CustomClassifierPipeline:
    def __init__(self, model_path):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model.eval()  # Set model to evaluation mode

    def classify_batch(self, sequences):
        # Tokenize the input sequences
        inputs = self.tokenizer(sequences, return_tensors="pt", truncation=True, padding='max_length', max_length=128)

        # Disable gradient calculations for inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_class_ids = torch.argmax(probabilities, dim=1).tolist()

        return predicted_class_ids, probabilities.tolist()

# Path to the directory where the model and tokenizer are saved
model_path = "../model"
custom_classifier = CustomClassifierPipeline(model_path)

# Load test dataset
df = pd.read_csv('../data/test_log_data.csv')  # Replace with your test data file

# Map labels to integers
label_mapping = {"Anomaly": 0, "Normal": 1}
df['label'] = df['label'].map(label_mapping)

# Extract log entries and true labels
log_entries = df['log'].tolist()
true_labels = df['label'].tolist()

# Perform batch classification
pred_labels, _ = custom_classifier.classify_batch(log_entries)
print("true_labels",true_labels)
print("pred_labels",pred_labels)
# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, pred_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(true_labels, pred_labels, target_names=label_mapping.keys())
print("Classification Report:")
print(class_report)

# Accuracy
accuracy = accuracy_score(true_labels, pred_labels)
print("Accuracy:", accuracy)

# Save results to a file
with open('../data/classification_results.txt', 'w') as f:
    f.write("\nTime : "+str(datetime.datetime.now()))
    f.write("\ntrue_labels"+str(true_labels))
    f.write("\npred_labels"+str(pred_labels))
    f.write("\nConfusion Matrix:\n")
    f.write(str(conf_matrix) + "\n\n")
    f.write("Classification Report:\n")
    f.write(class_report + "\n")
    f.write("Accuracy: " + str(accuracy) + "\n")
