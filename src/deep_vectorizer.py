import os
import pickle
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, LocallyConnected2D, Flatten, Dense, Dropout

def get_deepface_model(input_shape = (152, 152, 3), num_classes = 8631):
    deepface = Sequential(name = 'DeepFace')
    deepface.add(Convolution2D(32, (11, 11), activation = relu, name = 'C1', input_shape = input_shape))
    deepface.add(MaxPooling2D(pool_size = 3, strides = 2, padding = 'same', name = 'M2'))
    deepface.add(Convolution2D(16, (9, 9), activation = relu, name = 'C3'))
    deepface.add(LocallyConnected2D(16, (9, 9), activation = relu, name = 'L4'))
    deepface.add(LocallyConnected2D(16, (7, 7), strides = 2, activation = relu, name = 'L5'))
    deepface.add(LocallyConnected2D(16, (5, 5), activation = relu, name = 'L6'))
    deepface.add(Flatten(name = 'F0'))
    deepface.add(Dense(4096, activation = relu, name = 'F7'))
    deepface.add(Dropout(rate = 0.5, name = 'D0'))
    deepface.add(Dense(num_classes, activation = softmax, name = 'F8'))
    
    return deepface

def load_deepface_weights(model, weights_path):
    model.load_weights(weights_path)
    
    return

def prep_for_deepface(image):
    return tf.expand_dims(tf.cast(image, tf.float32) / 255.0, axis = 0)

def get_deepface_vectorizer(weights_path, input_shape = (152, 152, 3), num_classes = 8631):
    model = get_deepface_model(input_shape = input_shape, num_classes = num_classes)
    load_deepface_weights(model, weights_path)
    
    model = Model(inputs = model.layers[0].input, outputs = model.layers[-3].output)
    
    dummy_arr = np. zeros((152, 152, 3))
    
    model.predict(prep_for_deepface(dummy_arr))
    
    return model

def vectorize_image(image, vectorizer):
    input_image = prep_for_deepface(image)
    
    return vectorizer.predict(input_image)[0]

def vectorize_processed_dataset(dataset_path, destination_root, path_suffix):
    
    # Initialize a vectorizer
    vectorizer = get_deepface_vectorizer('./pretrained_models/VGGFace2_DeepFace_weights_val-0.9034.h5')
    
    # Make sure the destination folder exists
    if not os.path.exists(destination_root):
        os.makedirs(destination_root)
    
    for i, image_name in enumerate(os.listdir(dataset_path)):
        image_path = os.path.join(dataset_path, image_name)
        print(f'Vectorizing image {i + 1} from {image_path}')
        vector_path = os.path.join(destination_root, image_name.split('.')[0] + f'-{path_suffix}.pickle')

        # If the vector already exists (possibly from a prior session), skip
        if os.path.exists(vector_path):
            print(f'Vector {i + 1} already exists as {vector_path}')
            continue

        # Get vector and write to disk
        image = cv2.imread(image_path)
        vector = vectorize_image(image, vectorizer)

        with open(vector_path, 'wb') as f:
            print(f'Pickling vector {i + 1} at {vector_path}')
            pickle.dump(vector, f)

    return
     