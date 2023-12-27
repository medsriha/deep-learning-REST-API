import numpy as np
from keras.preprocessing.image import img_to_array
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from threading import Thread
from PIL import Image
import base64
import flask
import redis
import uuid
import time
import json
import sys
import io

SERVER_SLEEP = 0.25
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNEL = 3
IMAGE_DTYPE = "float32"
IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 32
CLIENT_SLEEP = 0.25

app = flask.Flask("__name__")
db = redis.StrictRedis(host="localhost", port=6379, db=0)
model = None

def base64_encode_image(data):
    """
    Encode a NumPy array as a base64 string.

    Parameters:
        data (np.ndarray): The NumPy array to be encoded.

    Returns:
        str: The base64-encoded string.
    """
    return base64.b64encode(data).decode("utf-8")

def base64_decode_image(data, dataType, shape):
    """
    Decode a base64-encoded string into a NumPy array.

    Parameters:
        data (str): The base64-encoded string.
        dataType (str): The datatype of the NumPy array.
        shape (tuple): The shape of the NumPy array.

    Returns:
        np.ndarray: The decoded NumPy array.
    """
    if sys.version_info.major == 3:
        data = bytes(data, encoding="utf-8")
    data = np.frombuffer(base64.decodebytes(data), dtype=dataType)
    data = data.reshape(shape)
    return data

def prepare_image(image, target):
    """
    Prepare an image for classification.

    Parameters:
        image (PIL.Image.Image): The input image.
        target (tuple): The target size of the image.

    Returns:
        np.ndarray: The prepared image as a NumPy array.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    return image

def classify_process():
    """
    Continuously poll for new images to classify and update the database with predictions.
    """
    print("* Loading model...")
    global model
    model = ResNet50(weights="imagenet")
    print("* Model loaded")

    while True:
        queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
        imageIDs = []
        batch = None

        for q in queue:
            q = json.loads(q.decode("utf-8"))
            image = base64_decode_image(q["image"], IMAGE_DTYPE, (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))

            if batch is None:
                batch = image
            else:
                batch = np.vstack([batch, image])

            imageIDs.append(q["id"])

        if len(imageIDs) > 0:
            print("* Batch size: {}".format(batch.shape))
            preds = model.predict(batch)
            results = imagenet_utils.decode_predictions(preds)

            for (imageID, resultSet) in zip(imageIDs, results):
                output = []

                for (imagenetID, label, prob) in resultSet:
                    r = {"label": label, "probability": float(prob)}
                    output.append(r)

                db.set(imageID, json.dumps(output))

            db.ltrim(IMAGE_QUEUE, len(imageIDs), -1)

        time.sleep(SERVER_SLEEP)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint for image prediction.
    """
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image = image.copy(order="C")

            k = str(uuid.uuid4())
            d = {"id": k, "image": base64_encode_image(image)}
            db.rpush(IMAGE_QUEUE, json.dumps(d))

            while True:
                output = db.get(k)

                if output is not None:
                    output = output.decode("utf-8")
                    data["predictions"] = json.loads(output)
                    db.delete(k)
                    break

                time.sleep(CLIENT_SLEEP)

            data["success"] = True

    return flask.jsonify(data)

if __name__ == "__main__":
    print("* Starting model service...")
    t = Thread(target=classify_process, args=())
    t.daemon = True
    t.start()

    print("* Starting web service...")
    app.run(debug=True)
