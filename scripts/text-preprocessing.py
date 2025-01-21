import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from scipy import stats

# Load text data from CSV file
text_data_path = 'text_data.csv'  # Replace with actual path to your text data CSV file
df = pd.read_csv(text_data_path)

# Display columns for reference
print(df.columns)

# Handle missing values in text data (replace NaN with empty strings or a placeholder)
df['Message'] = df['Message'].fillna('')  # Replace NaN values with empty strings
df['Message'] = df['Message'].astype(str)  # Ensure all values are strings

# Handle missing values in numerical columns (if applicable)
numerical_columns = ['ID']  # 'Date' is a datetime, no need to handle missing for now
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())  # Fill with median for numerical columns

# Detect and handle outliers in numerical columns (if applicable)
def detect_outliers(df, column_name):
    # Calculate the Z-score for the column
    z_scores = np.abs(stats.zscore(df[column_name]))
    return df[z_scores < 3]  # Keep rows where z-score is less than 3 (assuming normal distribution)

# Apply outlier detection to numerical columns (e.g., 'ID')
for column in numerical_columns:
    df = detect_outliers(df, column)

# Extract the 'Message' column as a list
text_data = df['Message'].tolist()

# Tokenization and padding
max_words = 10000
max_sequence_length = 100
embedding_dim = 50

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(text_data)
text_sequences = tokenizer.texts_to_sequences(text_data)
text_padded = pad_sequences(text_sequences, maxlen=max_sequence_length, padding='post')

# Save the processed text data to a new CSV file
processed_text_data_path = 'processed_text_data.csv'
processed_df = pd.DataFrame(text_padded)
processed_df.to_csv(processed_text_data_path, index=False)

print(f"Processed text data saved to {processed_text_data_path}")
