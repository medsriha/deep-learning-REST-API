# USAGE
# python simple_request.py

# import the necessary packages
import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
import urllib.request

url = "http://www.royalcanin.ca/~/media/Royal-Canin-Canada/Product-Categories/cat-adult-landing-hero.ashx"

try:
	req = urllib.request.urlretrieve(url, "local-filename.jpg")
except:
	print('Image not available')

IMAGE_PATH = "cat.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

if r == 200:
	# loop over the predictions and display them
	for (i, result) in enumerate(r["predictions"]):
		print("{}. {}: {:.4f}".format(i + 1, result["label"],
			result["probability"]))

# otherwise, the request failed
else:
	print("Request failed")
