import requests

destination = 'file.zip'

# Download the file
url = f"https://drive.google.com/file/d/1mmSihg4zxrabWEDY0W7Cs-JY1tW7OKpl/view?usp=drive_link"
response = requests.get(url, stream=True)
with open(destination, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

# Extract the downloaded zip file
import zipfile
with zipfile.ZipFile(destination, 'r') as zip_ref:
    zip_ref.extractall('extracted')
