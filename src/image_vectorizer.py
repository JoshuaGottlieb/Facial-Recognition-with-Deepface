import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from image_preprocessor import ImagePreprocessor

from tensorflow.keras.layers import LocallyConnected2D

class ImageVectorizer:
    def __init__(self, weights_path = './VGGFace2_DeepFace_weights_val-0.9034.h5'):
        self.weights_path = weights_path
        self.model = None

    def _get_deepface_graph(self, input_shape = (152, 152, 3), num_classes = 8631):
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
        
        self.model = deepface
        return

    def _load_model(self):
        self.model.load_weights(self.weights_path)
        return

    def _prep_for_deepface(self, image):
        return tf.expand_dims(tf.cast(image, tf.float32) / 255.0, axis = 0)

    def initialize(self, input_shape = (152, 152, 3), num_classes = 8631):
        self._get_deepface_graph(input_shape = input_shape, num_classes = num_classes)
        self._load_model()

        self.model = Model(inputs = self.model.layers[0].input, outputs = self.model.layers[-3].output)

        dummy_arr = np.zeros((152, 152, 3))

        self.model.predict(self._prep_for_deepface(dummy_arr), verbose = 0)
        return

    def vectorize_image(self, image_path, preprocess = True, preprocess_type = 'normal',
                        input_shape = (152, 152, 3), num_classes = 8631, **kwargs):
        if self.model == None:
            self.initialize(input_shape = input_shape, num_classes = num_classes)
            
        image_wrapper = ImagePreprocessor(image_path)
        
        if preprocess:
            if preprocess_type == 'normal':
                image_wrapper.preprocess_image(**kwargs)
                input_image = self._prep_for_deepface(image_wrapper.get_resized_image())
            elif preprocess_type == 'ghosh':
                image_wrapper.preprocess_ghosh()
                input_image = self._prep_for_deepface(image_wrapper.get_resized_image())
            else:
                print(f"Unable to determine preprocessing type.\n"
                      + "Valid preprocessing types are 'normal' and 'ghosh'. Using unprocessed image.")
                input_image = self._prep_for_deepface(image_wrapper.get_image())
        else:
            input_image = self._prep_for_deepface(image_wrapper.get_image())

        return self.model.predict(input_image)[0]