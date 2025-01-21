import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from scipy import stats
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True, help='text_data.csv')
args = parser.parse_args()

# Define base directory
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current script directory

# Define input and output paths
input_path = os.path.join(base_dir, 'Amharic-NER', args.input_file)  # Input file in Amharic-NER project folder
output_dir = os.path.join(base_dir, 'Amharic-NER', 'data')  # Output folder in the 'Amharic-NER/data' folder

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define output file path
output_file = os.path.join(output_dir, 'proces_text_data.csv')

# Load text data from CSV file
df = pd.read_csv(input_path)

# Display columns for reference
print(f"Columns in the dataset: {df.columns.tolist()}")

# Handle missing values in text data (replace NaN with empty strings)
df['Message'] = df['Message'].fillna('').astype(str)  # Ensure all values are strings

# Handle missing values in numerical columns (if applicable)
numerical_columns = ['ID']  # Add other numerical columns as needed
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

# Detect and handle outliers in numerical columns (using IQR or Z-scores)
def detect_outliers_iqr(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

# Apply outlier detection to numerical columns
for column in numerical_columns:
    df = detect_outliers_iqr(df, column)

# Extract the 'Message' column as a list for tokenization
text_data = df['Message'].tolist()

# Tokenization parameters
max_words = 10000
max_sequence_length = 100

# Initialize and fit the tokenizer on the text data
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(text_data)
text_sequences = tokenizer.texts_to_sequences(text_data)

# Pad the sequences to ensure they are of the same length
text_sequences = pad_sequences(text_sequences, maxlen=max_sequence_length, padding='post')

# Display the shape of the padded sequences
print(f"Shape of the processed text sequences: {text_sequences.shape}")

# Calculate and print the vocabulary size
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token
print(f"Vocabulary size: {vocab_size}")

# Save the processed text data to a new CSV file
processed_df = pd.DataFrame(text_sequences)
processed_df.to_csv(output_file, index=False)

print(f"Processed text data saved to {output_file}")

# Optional: Save the tokenizer for later use
tokenizer_json = tokenizer.to_json()
with open(os.path.join(base_dir, 'Amharic-NER', 'tokenizer.json'), 'w') as json_file:
    json_file.write(tokenizer_json)

print("Tokenizer saved to 'Amharic-NER/tokenizer.json'")
