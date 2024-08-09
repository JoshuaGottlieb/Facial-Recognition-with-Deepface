import firebase_admin
from firebase_admin import credentials, firestore
import uuid
from face_api import get_image_vector
import os
import tensorflow as tf
import numpy as np
import shutil

print(tf.__version__)

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

# Initialize Firestore DB
db = firestore.client()

# Path to the folder containing images
image_folder = '../data/dummy_dataset/images_to_vectorize'
destination_folder = '../data/dummy_dataset/input_images'

# Reference to your Firestore collection
collection_ref = db.collection('vectors')

# Loop through each image in the folder
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    print('image_path', image_path)

    # Get the image vector
    image_vector = get_image_vector(image_path)

    # Convert numpy array to list
    image_vector_list = image_vector.tolist()

    # Generate a unique ID for the image
    unique_id = str(uuid.uuid4())

    # Create a dictionary to store in Firestore
    data = {
        image_name+unique_id: image_vector_list,
    }

    # Add the data to Firestore
    collection_ref.document(unique_id).set(data)

    # Move the processed image to the destination folder
    shutil.move(image_path, os.path.join(destination_folder, image_name))

print("All images have been processed and stored in Firestore.")