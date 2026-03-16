import requests
import base64
import time
import random
import json


def send_image(image_path, json_path):
    with open(image_path, "rb") as img_file:
        # Convert image to Base64 string
        b64_string = base64.b64encode(img_file.read()).decode('utf-8')

    with open(json_path, "rb") as json_file:
        data = json.load(json_file)

    data["base64"] = b64_string
    payload = data

    response = requests.post("http://localhost:4000/api/push-data", json=payload)
    print(f"Status: {response.status_code}, Response: {response.text}")


# Example usage
for x in range(4):
    n = random.randint(0, 14)
    send_image(f"test_images/image{n}.png", "test_data.json")
