import os
import numpy as np
import cv2
import dlib
import mediapipe as mp
from retinaface import RetinaFace
from . import face_utils as ft
from . import utils

class ImagePreprocessor:
    def __init__(self, image_path, shape_predictor_path = './pretrained_models/shape_predictor_5_face_landmarks.dat'):
        self.image_path = image_path
        self.image = None
        self.aligned_image = None
        self.cropped_image = None
        self.resized_image = None
        self.a_backend = None
        self.c_backend = None
        self.dlib_details = None
        self.rf_details = None
        self.mp_details = None
        self.resize_dim = None
        self.pad = None
        self.pad_color = None
        self.error_code = None
        self.error_string = None
        self.facial_region = None
        self.shape_predictor_path = shape_predictor_path
        
        return

    def _clear_attributes(self):
        self.image = None
        self.aligned_image = None
        self.cropped_image = None
        self.resized_image = None
        self.a_backend = None
        self.c_backend = None
        self.dlib_details = None
        self.rf_details = None
        self.mp_details = None
        self.resize_dim = None
        self.pad = None
        self.pad_color = None
        self.error_code = None
        self.error_string = None
        self.facial_region = None
        return
    
    def _clear_backend_details(self):
        self.dlib_details = None
        self.rf_details = None
        self.mp_details = None
        return
    
    def _clear_error_code(self):
        self.error_code = None
        return
        
    def load_image(self):
        self.image = cv2.imread(self.image_path)
        self.aligned_image = self.image.copy()
        self.cropped_image = self.image.copy()
        self.resized_image = self.image.copy()
        return
        
    def get_image(self):
        self.load_image()
        return self.image
    
    def get_aligned_image(self):
        return self.aligned_image

    def get_cropped_image(self):
        return self.cropped_image
    
    def get_resized_image(self):
        return self.resized_image
    
    def _set_error_string(self):
        pass
    
    def get_error(self):
        return self.error_code, self.error_string
    
    def _get_dlib_faces(self, image):
        """
        Uses dlib's face landmark shape predictor to detect faces in an image.
        Details about the centermost face in the image are returned if one or more faces are found, else returns -1.

        args:
            image: numpy array representing the input image
            shape_predictor_path: string representing the path to the pretrained shape predictor landmark weights

        return: dlib chip details object containing information about the centermost face found in the image;
                if no faces are found, returns -1 for use in further error handling
        """
        # Default shape predictor path: './pretrained_models/shape_predictor_5_face_landmarks.dat'

        # Instantiate a dlib face detector and shape predictor
        detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(self.shape_predictor_path)

        # Get faces from image
        detections = detector(image, 0)
        faces = dlib.full_object_detections()
        for detection in detections:
            faces.append(shape_predictor(image, detection))

        chip_details = dlib.get_face_chip_details(faces)

        chip_index = 0

        # If no faces are found, return -1.
        if len(chip_details) == 0:
            self.error_code = -1
            return

        # If more than one face is found, get the index of the centermost face.
        if len(chip_details) > 1:
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            face_centers = [(c.rect.center().x, c.rect.center().y) for c in chip_details]
            chip_index = ft.get_index_center_face(image_center, face_centers)

        self.dlib_details = chip_details[chip_index]

        return
    
    def _get_retinaface_faces(self, image):
        """
        Uses RetinaFace to detect faces in an image.
        Details about the centermost face in the image are returned if one or more faces are found, else returns -1.

        args:
            image: numpy array representing the input image

        return: dictionary representing facial data of centermost face;
                if no faces are found, returns -1 for further error handling
        """
        faces = RetinaFace.detect_faces(img_path = image)

        face_key = 'face_1'

        # If no faces are found, return -1.
        if len(faces.keys()) == 0:
            self.error_code = -1
            return

        # If multiple faces are found, get the dictionary key for the centermost face.
        if len(faces.keys()) > 1:
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            face_noses = [(faces[key]['landmarks']['nose'][0], faces[key]['landmarks']['nose'][1]) for key in faces.keys()]
            face_index = ft.get_index_center_face(image_center, face_noses)
            face_key = list(faces.keys())[face_index]

        self.rf_details = faces[face_key]

        return

    def _get_mediapipe_faces(self, image):
        """
        Uses MediaPipe to detect faces in an image.
        Details about the centermost face in the image are returned if one or more faces are found, else returns -1.

        args:
            image: numpy array representing the input image

        return: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList representing
                the found landmarks for the centermost face; if no faces are found, returns -1 for further error handling
        """
        mp_face_mesh = mp.solutions.face_mesh

        with mp_face_mesh.FaceMesh(static_image_mode = True,
                                   max_num_faces = 5,
                                   refine_landmarks = True) as face_mesh:
            results = face_mesh.process(image[:,:,::-1])

        results_key = 0

        # If no faces are found, return -1.
        if results.multi_face_landmarks is None:
            self.error_code = -1
            return

        # If multiple faces are found, get the index of the centermost face.
        if len(results.multi_face_landmarks) > 1:
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            # Landmarks 475 and 470 are points roughly near the center of each eye.
            # Rough face centers are calculated by taking the average of the x- and y- coordinates of these landmarks.
            rough_face_centers = [np.sum((np.array((face.landmark[475].x, face.landmark[475].y)),
                                          np.array((face.landmark[470].x, face.landmark[470].y))), axis = 0) / 2\
                                 for face in results.multi_face_landmarks]
            # MediaPipe uses image normalized coordinates, so they must be converted back to pixel coordinates.
            rough_face_centers_pixel = [ft.normal_to_pixel(f[0], f[1], image.shape[1], image.shape[0]) for f in
                                        rough_face_centers]
            results_key = ft.get_index_center_face(image_center, rough_face_centers_pixel)

        self.mp_details = results.multi_face_landmarks[results_key]
        return
    
    def _dlib_align(self):
        """
        Uses rectanglular and angular data from a dlib chip details object to rotate an image.

        args:
            image: numpy array representing the input image
            chip_detail: dlib chip details object containing face data

        return: numpy array representing the rotated image
        """    
        face_center = (self.dlib_details.rect.center().x, self.dlib_details.rect.center().y)
        self.aligned_image = ft.rotate_from_angle(self.image, face_center, self.dlib_details.angle, 1, degrees = False)

        return
    
    def _retinaface_align(self):
        """
        Uses facial data from RetinaFace to rotate an image.

        args:
            image: numpy array representing the input image
            face_data: dictionary with facial data

        return: numpy array representing the rotated image
        """    
        right_eye = np.array(self.rf_details['landmarks']['right_eye'])
        left_eye = np.array(self.rf_details['landmarks']['left_eye'])

        angle, direction = ft.get_angle_from_eyes(left_eye, right_eye)

        self.aligned_image = ft.rotate_from_angle(self.image, tuple(np.array(self.image.shape[1::-1]) / 2), angle, direction)

        return
    
    def _mediapipe_align(self):
        """
        Uses facial landmarks from MediaPipe to rotate an image.

        args:
            image: numpy array representing the input image
            mp_results: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList representing
                        the found landmarks for the face

        return: numpy array representing the rotated image
        """
        # Landmarks for the left and right irises of the face.
        # MediaPipe does not have dedicated landmarks for the centers of each eye.
        left_iris = [474,475, 476, 477]
        right_iris = [469, 470, 471, 472]
        landmarks = [left_iris, right_iris]

        pixels = []
        for eye in landmarks:
            eye_pixels = []
            for num in eye:
                landmark = self.mp_details.landmark[num]
                coords = ft.normal_to_pixel(float(landmark.x), float(landmark.y), self.image.shape[1], self.image.shape[0])
                eye_pixels.append(np.array(coords))
            pixels.append(np.array(eye_pixels))

        # Derive rough eye centers by averaging pixel coordinates from landmarks.
        eye_centers = [(np.sum(eye, axis = 0) / len(eye)).astype(int) for eye in pixels]

        angle, direction = ft.get_angle_from_eyes(eye_centers[0], eye_centers[1])

        self.aligned_image = ft.rotate_from_angle(self.image, tuple(np.array(self.image.shape[1::-1]) / 2), angle, direction)

        return
    
    def _get_dlib_region(self):
        """
        Uses rectanglular data from a dlib chip details object to extract x, y coordinate pairs representing a facial region.

        args:
            chip_detail: dlib chip details object containing face data

        return: (x_min, x_max, y_min, y_max) tuple of ints representing facial region for use in numpy slicing
        """
        # Dlib's coordinate system has y = 0 at the top of the image, the opposite of numpy
        # Thus, y_min is the top of the rectangle, while y_max, is the bottom of the rectangle
        x_min = int(np.round(self.dlib_details.rect.left()))
        x_max = int(np.round(self.dlib_details.rect.right()))

        y_min = int(np.round(self.dlib_details.rect.top()))
        y_max = int(np.round(self.dlib_details.rect.bottom()))
        
        self.facial_region = x_min, x_max, y_min, y_max

        return
    
    def _get_retinaface_region(self):
        """
        Uses facial data from RetinaFace to extract the facial region.

        args:
            face_data: dictionary with facial data

        return: (x_min, x_max, y_min, y_max) tuple of ints representing facial region for use in numpy slicing
        """
        self.facial_area = self.rf_details['facial_area']
        return
    
    def _get_mediapipe_region(self, image):
        """
        Uses facial landmarks from MediaPipe to extract the facial region.

        args:
            image: numpy array of the image, used to convert landmark coordinates to pixel form
            mp_results: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList representing
                        the found landmarks for the face

        return: (x_min, x_max, y_min, y_max) tuple of ints representing facial region for use in numpy slicing
        """
        # Calculate the bounding box for the face based off of the minimum and maximum coordinates in found landmarks.
        # Landmarks are in image normalized coordinates, so must be converted back to pixel coordinates.
        x_min = np.clip(0, int(np.round(min([k.x for k in self.mp_details.landmark]) * image.shape[1])), image.shape[1])
        x_max = np.clip(0, int(np.round(max([k.x for k in self.mp_details.landmark]) * image.shape[1])), image.shape[1])
        y_min = np.clip(0, int(np.round(min([k.y for k in self.mp_details.landmark]) * image.shape[0])), image.shape[0])
        y_max = np.clip(0, int(np.round(max([k.y for k in self.mp_details.landmark]) * image.shape[0])), image.shape[0])

        self.facial_area = x_min, x_max, y_min, y_max
        return
    
    def _align_image(self, backend = ['dlib', 'retinaface', 'mediapipe']):
        """
        Uses designated backends to align an image along the centermost face.

        args:
            image: numpy array representing the input image
            backend: str or list of designated backends to use for aligning images in order of preference.
                     defaults to ['dlib', 'retinaface', 'mediapipe']

        return: numpy array representing the rotated image, and the used backend as (rotated image, backend);
                if no valid backends are submitted or backend is an invalid type, returns -1;
                if no backends are able to detect faces, returns -1
        """

        self._clear_backend_details()
        self._clear_error_code()
        
        # Format and confirm backend parameter
        backend = utils.confirm_valid_list_input(backend, valid_options = ['dlib', 'retinaface', 'mediapipe'])

        # Handling error code from backend confirmation, exiting due to invalid backend input
        if type(backend) == int:
            self.error_code = -1
            return

        # Loop through backends, when one succeeds, return the rotated image and the backend used.
        for b in backend:
            if b == 'dlib':
                self._get_dlib_faces(self.image)
                if self.dlib_details is None:
                    continue
                self._dlib_align()
                self.a_backend = b
                return

            if b == 'retinaface':
                self._get_retinaface_faces(self.image)
                if self.rf_details is None:
                    continue
                self._retinaface_align()
                self.a_backend = b
                return

            if b == 'mediapipe':
                self._get_mediapipe_faces(self.image)
                if self.mp_details is None:
                    continue            
                self._mediapipe_align()
                self.a_backend = b
                return

        # If no backends found faces, return -1.
        self.error_code = -1
        return
    
    def _crop_image(self, backend = ['mediapipe', 'retinaface', 'dlib']):
        """
        Uses designated backends to crop an image around the centermost face.

        args:
            image: numpy array representing the input image
            backend: str or list of designated backends to use for cropping images in order of preference.
                     defaults to ['mediapipe', 'retinaface', 'dlib']

        return: numpy array representing the cropped image, and the used backend as (cropped image, backend);
                if no valid backends are submitted or backend is an invalid type, returns -1;
                if no backends are able to detect faces, returns -1
        """
        self._clear_backend_details()
        self._clear_error_code()
        
        # Format and confirm backend parameter
        backend = utils.confirm_valid_list_input(backend, valid_options = ['mediapipe', 'retinaface', 'dlib'])

        # Handling error code from backend confirmation, exiting due to invalid backend input
        if type(backend) == int:
            self.error_code = -1
            return

        # Loop through backends, when one succeeds, return the cropped image and the backend used.            
        for b in backend:
            if b == 'mediapipe':
                self._get_mediapipe_faces(self.aligned_image)
                if self.mp_details is None:
                    continue
                self._get_mediapipe_region(self.aligned_image)
                self.cropped_image = self.aligned_image[self.facial_area[2]:self.facial_area[3],
                                           self.facial_area[0]:self.facial_area[1], :]
                self.c_backend = b
                return

            if b == 'retinaface':
                self._get_retinaface_faces(self.aligned_image)
                if self.rf_details is None:
                    continue            
                self._get_retinaface_region()
                self.cropped_image = self.aligned_image[self.facial_area[2]:self.facial_area[3],
                                           self.facial_area[0]:self.facial_area[1], :]
                self.c_backend = b
                return

            if b == 'dlib':
                self._get_dlib_faces(self.aligned_image, self.shape_predictor_path)
                if self.dlib_details is None:
                    continue            
                self.get_dlib_region()
                self.cropped_image = self.aligned_image[self.facial_area[2]:self.facial_area[3],
                                           self.facial_area[0]:self.facial_area[1], :]
                self.c_backend = b
                return

        # If no backends found faces, return -1.   
        self.error_code = -1
        return
    
    def _resize_image(self):
        """
        Resizes an image to desired end dimensions.
        Compresses on axes where image shape is larger than end dimension.
        Pads on axes images shape is smaller than end dimension, using specified padding color.

        args:
            image: numpy array representing the input image
            end_dim: (width, height) array-like representing numpy slice-form end dimensions
            pad: Whether to pad the image. If True, applies padding using color, otherwise uses interpolation for upscaling
            color: scalar or array-like to pass to np.full for padding image, only used if pad is True;
                   default (0, 0, 0), tuples are in BGR format from OpenCV.

        return: numpy array representing resized image
        """
        self.resized_image = self.cropped_image.copy()

        if self.pad:
            # First, compress image along axes bigger than end dimension using cv2.resize with area interpolation.
            if self.resized_image.shape[0] > self.resize_dim[0] or self.resized_image.shape[1] > self.resize_dim[1]:
                self.resized_image = cv2.resize(self.resized_image, np.minimum(self.resized_image.shape[0:-1],
                                                                               self.resize_dim)[::-1],
                                                interpolation = cv2.INTER_AREA)

            # Second, pad image using specified color.
            # We want to pad second so that the interpolation does not pick up the padded data.
            if self.resized_image.shape[0] < self.resize_dim[0] or self.resized_image.shape[1] < self.resize_dim[0]:
                old_height, old_width, channels = self.resized_image.shape
                new_height, new_width = np.maximum(self.resized_image.shape[0:-1], self.resize_dim)
                x_center = (new_width - old_width) // 2
                y_center = (new_height - old_height) // 2

                y_top = y_center + old_height
                x_right = x_center + old_width

                padded_image = np.full((new_height, new_width, channels), self.pad_color, dtype = np.uint8)

                padded_image[y_center:y_top, x_center:x_right] = self.resized_image

                self.resized_image = padded_image.copy()
        else:
            if self.resized_image.shape[0] != self.resize_dim[0] or self.resized_image.shape[1] != self.resize_dim[1]:
                # If purely shrinking, use cv2.INTER_AREA, else, use cv2.INTER_CUBIC
                if self.resized_image.shape[0] > self.resize_dim[0] and self.resized_image.shape[1] > self.resize_dim[1]:
                    interpolation = cv2.INTER_AREA
                else:
                    interpolation = cv2.INTER_CUBIC
                self.resized_image = cv2.resize(self.resized_image, self.resize_dim[::-1], interpolation = interpolation)

        return
    
    def preprocess_image(self, alignment_backend = ['dlib', 'retinaface', 'mediapipe'],
                         crop_backend = ['mediapipe', 'retinaface', 'dlib'],
                         end_dim = (152, 152), color = (0, 0, 0), pad = True,
                         steps = ['align', 'crop', 'resize']):
    
        if type(steps) == str:
            if steps == 'full':
                steps = ['align', 'crop', 'resize']
            else:
                steps = [steps]

        steps = utils.confirm_valid_list_input(steps, valid_options = ['align', 'crop', 'resize'])

        if type(steps) == int:
            self.error_code = -1
            return
        
        self._clear_attributes()
        self.load_image()
        self.resize_dim = end_dim
        self.pad = pad
        self.pad_color = color

        for step in steps:
            self._clear_error_code()
            if step == 'align':
                self._align_image(backend = alignment_backend)
                if self.a_backend is None:
                    if 'crop' not in steps:
                        self.error_code = -1
                        return
                    else:
                       # Checks if alignment backend and crop backend are the same
                        # If all backends fail for alignment, they will fail for backend, so exit
                        if sorted(alignment_backend) == sorted(crop_backend):
                            self.error_code = -1
                            return
                continue

            if step == 'crop':
                self._crop_image(backend = crop_backend)
                if self.c_backend is None:
                    self.error_code = -1
                    return
                continue

            if step == 'resize':
                self._resize_image()
                continue
                
        return 

    def preprocess_ghosh(self):
        self._clear_attributes()
        self.load_image()
        self._get_dlib_faces(self.image)
        if self.dlib_details is None:
            self.error_code = -1
            return
        
        self.cropped_image = dlib.extract_image_chip(self.image, self.dlib_details)
        self.resize_dim = (152, 152)
        self.pad = True
        self.pad_color = (0, 0, 0)
        self._resize_image()
        return