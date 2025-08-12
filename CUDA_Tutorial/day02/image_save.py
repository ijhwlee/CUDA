import requests

# URL of the image
url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/1600px-Cute_dog.jpg?20140729055059'
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# Send a GET request to the URL
response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Open a file in binary write mode and save the image
    with open("downloaded_image.jpg", "wb") as f:
        f.write(response.content)
    print("Image downloaded and saved successfully.")
else:
    print(f"Failed to download image. Status code: {response.status_code}")
