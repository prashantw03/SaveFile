import pickle

# Load from a .pkl file
with open('vocab.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

# print(loaded_data)
# Print the first 5 elements if the data is a list or similar iterable
print(loaded_data)

# Alternatively, print the first few key-value pairs
# for i, (key, value) in enumerate(loaded_data.items()):
#     if i < 5:  # Adjust the number as needed
#         print(f"{key}: {value}")
#     else:
#         break
