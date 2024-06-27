import pandas as pd
import re
import os

# Define the start and end line indices
start_index = 0 # Replace with your actual start line index (0-based)
end_index = 200   # Replace with your actual end line index (0-based)

os.makedirs('../data', exist_ok=True)

# Read anomaly file into a DataFrame
anomaly_df = pd.read_csv('anomaly_label.csv', header=None)
anomaly_df.columns = ['BlockId', 'Label']

# Convert the DataFrame to a dictionary for quick lookup
anomaly_dict = pd.Series(anomaly_df.Label.values, index=anomaly_df.BlockId).to_dict()

# Prepare a list to hold the log lines and their labels
log_data = []
normal_log_data = []
anomaly_log_data = []

# Read the log file line by line
with open('HDFS.log', 'r') as log_file:
    for line_num, line in enumerate(log_file):
        # Only process lines within the specified range
        if start_index <= line_num <= end_index: #True: # // convert full file
            # Split the line by spaces
            fields = line.strip().split()
            key = False
            for field in fields:
                if re.match(r'^blk_', field):
                    key = field
                    break
            if key:
                print(key)
                # Determine the label (1 for anomaly, 0 for normal)
                label = anomaly_dict.get(key, 0)
                # Assign numerical label based on the condition
                if label == "Normal":
                    numerical_label = 1
                elif label == "Anomaly":
                    numerical_label = 0
                # Append the log line and its label to the list
                log_entry = [line.strip(), numerical_label]
                log_data.append(log_entry)

                if label == "Normal":
                    normal_log_data.append(log_entry)
                else:
                    anomaly_log_data.append(log_entry)

# Create DataFrames from the log data
log_df = pd.DataFrame(log_data, columns=['log', 'label'])
normal_log_df = pd.DataFrame(normal_log_data, columns=['log', 'label'])
anomaly_log_df = pd.DataFrame(anomaly_log_data, columns=['log', 'label'])

# Write the DataFrames to new CSV files
log_df.to_csv('../data/train_log_data.csv', index=False)
normal_log_df.to_csv('../data/train_normal_log_data.csv', index=False)
anomaly_log_df.to_csv('../data/train_anomaly_log_data.csv', index=False)
