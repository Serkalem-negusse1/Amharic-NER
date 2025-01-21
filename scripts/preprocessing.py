import os
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define text preprocessing function
def preprocess_text(text):
    """
    Preprocesses text by:
    - Lowercasing
    - Removing punctuation and special characters
    - Tokenizing
    - Removing stopwords
    - Lemmatizing words
    """
    # Lowercasing
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenization
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return " ".join(words)

# Define image preprocessing function
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocesses an image by:
    - Resizing the image
    - Normalizing pixel values (0-1 range)
    """
    # Load image
    image = load_img(image_path, target_size=target_size)
    
    # Convert image to array
    image_array = img_to_array(image)
    
    # Normalize pixel values to [0, 1]
    image_array = image_array / 255.0
    
    return image_array

# Example: Load and preprocess text data
text_file_path = 'telegram_data.csv'  # Replace with actual file path

# Check if the file exists and is not empty
if not os.path.exists(text_file_path):
    print(f"Error: The file {text_file_path} does not exist.")
    exit()

with open(text_file_path, 'r', encoding='utf-8') as file:
    text_data = file.readlines()

text_data = [text.strip() for text in text_data if text.strip()]  # Remove extra spaces or newlines and empty lines

# Ensure there is data to process
if len(text_data) == 0:
    print("Error: The text data file is empty.")
    exit()

# Preprocess text data
processed_text_data = [preprocess_text(text) for text in text_data]

# Ensure processed text data is not empty
if len(processed_text_data) == 0:
    print("Error: No valid text data after preprocessing.")
    exit()

# Example: Load and preprocess image data
image_dir = 'photos'  # Replace with actual image directory path

# Check if the image directory exists and contains images
if not os.path.exists(image_dir):
    print(f"Error: The directory {image_dir} does not exist.")
    exit()

# Get list of image files (JPEG, PNG)
image_files = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith(('.jpg', '.png'))]

# Ensure there are image files to process
if len(image_files) == 0:
    print("Error: No image files found in the directory.")
    exit()

# Preprocess images
processed_image_data = [preprocess_image(img_path) for img_path in image_files]

# Split the data into training and testing sets (80/20 split)
X_text_train, X_text_test = train_test_split(processed_text_data, test_size=0.2, random_state=42)
X_image_train, X_image_test = train_test_split(processed_image_data, test_size=0.2, random_state=42)

# Display a sample of preprocessed data
print("Processed Text Sample:", processed_text_data[:2])
print("Processed Image Data Shape Sample:", np.array(processed_image_data).shape)
