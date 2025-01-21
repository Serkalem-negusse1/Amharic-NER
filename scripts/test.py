from dotenv import load_dotenv
import os

# Load .env file from the 'scripts' directory
load_dotenv(r'e:\Amharic-NER\scripts\.env')

# Debug: Check if the environment variables are loaded correctly
api_id = os.getenv('API_ID')
api_hash = os.getenv('API_HASH')
phone = os.getenv('PHONE_NUMBER')

print(f"API_ID: {api_id}")
print(f"API_HASH: {api_hash}")
print(f"PHONE_NUMBER: {phone}")
