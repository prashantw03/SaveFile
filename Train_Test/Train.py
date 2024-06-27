import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

os.makedirs('../model', exist_ok=True)

# Load and preprocess data
def load_and_prepare_data(file_path):
    # Load data from CSV
    data = pd.read_csv(file_path)

    # Convert to Hugging Face dataset
    dataset = Dataset.from_pandas(data)

    # Tokenizer function
    def preprocess_function(examples):
        return tokenizer(examples['log'], truncation=True, padding='max_length', max_length=128)

    # Tokenize dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Split dataset into train and test (80-20 split)
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    return train_test_split

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Binary classification
tokenizer = BertTokenizer.from_pretrained(model_name)

# File path
file_path = "../data/train_log_data.csv"

# Load and prepare data
tokenized_dataset = load_and_prepare_data(file_path)

# Define the metric function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    f1 = f1_score(labels, predictions, average='binary')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Save the model
model.save_pretrained("../model")
tokenizer.save_pretrained("../model")
