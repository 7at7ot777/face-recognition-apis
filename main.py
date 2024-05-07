import os
import threading

from flask import Flask, request, jsonify
import face_recognition
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
app = Flask("__name__")


@app.route("/")
def index():
    return "Hello World"


# Initialize variables
known_face_encodings = []
known_face_names = []
@app.route('/train', methods=['POST'])
def train_model_route():
    training_data = request.json

    # Start a new thread for training to avoid blocking the main thread
    training_thread = threading.Thread(target=train_model, args=(training_data,))
    training_thread.start()

    return jsonify({'message': 'Training process started'})

def train_model(training_data):
    with app.app_context():
        if not training_data or not isinstance(training_data, list):
            return jsonify({'error': 'Invalid training data format'})

        for data in training_data:
            label = data.get('label')
            image_url = data.get('image_url')

            if not label or not image_url:
                continue  # Skip this iteration if label or image_url is missing

            # Send a GET request to download the image
            response = requests.get(image_url)

            if response.status_code != 200:
                continue  # Skip this iteration if failed to download the image

            # Save the image data to a temporary file
            with open('temp_image.jpg', 'wb') as f:
                f.write(response.content)

            # Load the image from the temporary file and encode the face
            image = face_recognition.load_image_file('temp_image.jpg')
            face_encodings = face_recognition.face_encodings(image)

            if not face_encodings:
                continue  # Skip this iteration if no face found in the image

            # Assuming only the first face is used for encoding
            face_encoding = face_encodings[0]

            # Add the face encoding and label to known_face_encodings and known_face_names lists
            known_face_encodings.append(face_encoding)
            known_face_names.append(label)
            print("Training " + label)

            # Remove the temporary file
            os.remove('temp_image.jpg')

        return jsonify({'message': 'Model trained successfully'})

# Endpoint to detect faces in an image and recognize the labels
@app.route('/detect', methods=['POST'])
def detect_faces():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'})

    image_file = request.files['image']

    # Load the image from URL
    image = face_recognition.load_image_file(image_file)

    # Find all the faces and face encodings in the image
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # Compare face encodings with known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the best match for the face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    return jsonify({'face_labels': face_names})


if __name__ == '__main__':
    app.run(debug=True)
