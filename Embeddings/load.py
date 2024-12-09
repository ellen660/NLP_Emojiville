import requests

# Replace FILE_ID with your file's ID
file_id = '<FILE_ID>'
destination = 'file.zip'

# Download the file
url = f"https://drive.google.com/uc?id={file_id}&export=download"
response = requests.get(url, stream=True)
with open(destination, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

# Extract the downloaded zip file
import zipfile
with zipfile.ZipFile(destination, 'r') as zip_ref:
    zip_ref.extractall('extracted')
