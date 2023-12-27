# USAGE
# python simple_request.py

# import the necessary packages
import requests
import urllib.request

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"

# URL of the image you want to classify
url = "http://www.royalcanin.ca/~/media/Royal-Canin-Canada/Product-Categories/cat-adult-landing-hero.ashx"

# Try to retrieve the image from the URL and save it locally
try:
    req = urllib.request.urlretrieve(url, "local-filename.jpg")
except:
    print('Image not available')

# Local path to the image to be classified
IMAGE_PATH = "local-filename.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# check if the request was successful (HTTP status code 200)
if r == 200:
    # loop over the predictions and display them
    for (i, result) in enumerate(r["predictions"]):
        print("{}. {}: {:.4f}".format(i + 1, result["label"],
              result["probability"]))

# otherwise, the request failed
else:
    print("Request failed")
