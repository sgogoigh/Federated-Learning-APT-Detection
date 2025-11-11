import zipfile
import os

# Path to your zip file
zip_path = 'results.zip'

# Directory where you want to extract the files
extract_path = os.getcwd()

# # Create the extraction directory if it doesn't exist
# os.makedirs(extract_path, exist_ok=True)

# Open the zip file in read mode
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Extract all the contents into the specified directory
    zip_ref.extractall(extract_path)

print(f"Successfully extracted all files to: {extract_path}")