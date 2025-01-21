import os
from keras.preprocessing.image import load_img, img_to_array, save_img
import numpy as np

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

# Example: Load and preprocess image data
image_dir = '/photos'  # Replace with actual image directory path
output_dir = '/processed_images'  # Directory where preprocessed images will be saved

# Check if the image directory exists
if not os.path.exists(image_dir):
    print(f"Error: The directory {image_dir} does not exist.")
    exit()

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get list of image files (JPEG, PNG)
image_files = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith(('.jpg', '.png'))]

# Ensure there are image files to process
if len(image_files) == 0:
    print("Error: No image files found in the directory.")
    exit()

# Preprocess and save images
for i, img_path in enumerate(image_files):
    processed_image = preprocess_image(img_path)
    
    # Generate the output image path
    output_image_path = os.path.join(output_dir, f"processed_image_{i+1}.png")
    
    # Save the preprocessed image to the output directory
    save_img(output_image_path, processed_image)
    print(f"Saved preprocessed image: {output_image_path}")

print(f"All images processed and saved to {output_dir}")
