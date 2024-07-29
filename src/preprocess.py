import os
import math
import numpy as np
import cv2
import dlib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from retinaface.commons.postprocess import rotate_facial_area
from retinaface import RetinaFace
from . import deep_vectorizer as dp

def normal_to_pixel(x, y, w, h, invert = False):
    """
    Converts a set of x, y coordinates from image normalized coordinates to standard pixel coordinates.
    
    args:
        x: x-coordinate value to convert
        y: y-coordinate value to convert
        w: width of image that the coordinates were drawn from
        y: height of image that the coordinates were drawn from
        invert: If False, convert from normalized coordinates to pixel coordinates.
                If True, convert from pixel coordinates to normalized coordinates.
    
    returns: list of converted coordinates in the form [x, y]
    """
    if invert:
        return list(np.array((x, y)) / np.array((w, h)))
    return list((np.array((x, y)) * np.array((w, h))).astype(int))

def get_angle_from_eyes(left_eye, right_eye, degrees = True):
    """
    Obtains the angle and direction of the line running through the (x, y) coordinates of two eyes.
    
    args:
        left_eye: (x, y) array-like containing pixel coordinates for the left eye
        right_eye: (x, y) array-like containing pixel coordinates for the right eye
        degree: If False, returns angle in radians. If True, returns angle in degrees.
        
    returns: angle between eyes and a direction (+1 or -1) for rotation instructions in the form (angle, direction)
    """
    
    # If the left and right eye have the same vertical y-values, the angle is zero.
    if right_eye[1] - left_eye[1] == 0:
        return 0, 1
    
    # Compute the cosine between the two coordinates using the dot product and norm definition of cosine.
    # Clips at plus or minus one for input into arccos function.
    cos_angle = np.clip(np.dot(left_eye,right_eye) / (np.linalg.norm(left_eye, 2) * np.linalg.norm(right_eye, 2)),
                        -1.0, 1.0)
    
    direction = math.copysign(1, (left_eye[1] - right_eye[1]))

    angle = np.arccos(cos_angle)
    if degrees:
        angle = np.rad2deg(angle)
        
    return angle, direction

def rotate_from_angle(image, center, angle, direction, degrees = True):
    """
    Rotates an image around specified center given an angle and direction.
    
    args:
        image: numpy array of image to be rotated
        center: (x, y) array-like containing pixel coordinates for the center around which to rotate
        angle: float or int representing the angle magnitude used to rotate
        direction: int representing the direction used to rotate;
                   values with magnitude != 1 will dilate or contract the resulting image
        degrees: If True, angle represents degrees. If False, angle represents radians.
        
    returns: numpy array representing the rotated image
    """
    if not degrees:
        angle = np.rad2deg(angle)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle * direction, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags = cv2.INTER_LINEAR)
    
    return rotated_image

def get_distance_center(image_center, face_centers):
    """
    Calculates the distance to the image center from given face centers.
    
    args:
        image_center: (x, y) array-like containing pixel coordinates of the center of the image
        face_centers: 2-D array-like containing (x, y) pixel coordinates of the centers of faces
        
    return: 2-D array-like containing absolute distances in (x, y) form
    """
    return np.sum(np.abs(np.array(image_center) - np.array(face_centers)), axis = 1)

def get_index_center_face(image_center, face_centers):
    """
    Calculates the index of the centermost face in an image from face_centers array.
    
    args:
        image_center: (x, y) array-like containing pixel coordinates of the center of the image
        face_centers: 2-D array-like containing (x, y) pixel coordinates of the centers of faces
        
    return: int, index of the face_centers array corresponding to the most central face in the image
    """
    return np.argmin(get_distance_center(image_center, face_centers))

def get_dlib_faces(image, shape_predictor_path):
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
    shape_predictor = dlib.shape_predictor(shape_predictor_path)
    
    # Get faces from image
    detections = detector(image, 0)
    faces = dlib.full_object_detections()
    for detection in detections:
        faces.append(shape_predictor(image, detection))
        
    chip_details = dlib.get_face_chip_details(faces)
    
    chip_index = 0
    
    # If no faces are found, return -1.
    if len(chip_details) == 0:
        return -1

    # If more than one face is found, get the index of the centermost face.
    if len(chip_details) > 1:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        face_centers = [(c.rect.center().x, c.rect.center().y) for c in chip_details]
        chip_index = get_index_center_face(image_center, face_centers)
        
    return chip_details[chip_index]

def dlib_align(image, chip_detail):
    """
    Uses rectanglular and angular data from a dlib chip details object to rotate an image.
    
    args:
        image: numpy array representing the input image
        chip_detail: dlib chip details object containing face data
        
    return: numpy array representing the rotated image
    """    
    face_center = (chip_detail.rect.center().x, chip_detail.rect.center().y)
    rotated_image = rotate_from_angle(image, face_center, chip_detail.angle, 1, degrees = False)
    
    return rotated_image

def get_dlib_region(chip_detail):
    """
    Uses rectanglular data from a dlib chip details object to extract x, y coordinate pairs representing a facial region.
    
    args:
        chip_detail: dlib chip details object containing face data
        
    return: (x_min, x_max, y_min, y_max) tuple of ints representing facial region for use in numpy slicing
    """
    # Dlib's coordinate system has y = 0 at the top of the image, the opposite of numpy
    # Thus, y_min is the top of the rectangle, while y_max, is the bottom of the rectangle
    x_min = int(np.round(chip_detail.rect.left()))
    x_max = int(np.round(chip_detail.rect.right()))

    y_min = int(np.round(chip_detail.rect.top()))
    y_max = int(np.round(chip_detail.rect.bottom()))
    
    return x_min, x_max, y_min, y_max

def get_retinaface_faces(image):
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
        return -1
    
    # If multiple faces are found, get the dictionary key for the centermost face.
    if len(faces.keys()) > 1:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        face_noses = [(faces[key]['landmarks']['nose'][0], faces[key]['landmarks']['nose'][1]) for key in faces.keys()]
        face_index = get_index_center_face(image_center, face_noses)
        face_key = list(faces.keys())[face_index]
    
    return faces[face_key]

def retinaface_align(image, face_data):
    """
    Uses facial data from RetinaFace to rotate an image.
    
    args:
        image: numpy array representing the input image
        face_data: dictionary with facial data
        
    return: numpy array representing the rotated image
    """    
    right_eye = np.array(face_data['landmarks']['right_eye'])
    left_eye = np.array(face_data['landmarks']['left_eye'])

    angle, direction = get_angle_from_eyes(left_eye, right_eye)
    
    rotated_image = rotate_from_angle(image, tuple(np.array(image.shape[1::-1]) / 2), angle, direction)

    return rotated_image

def get_retinaface_region(face_data):
    """
    Uses facial data from RetinaFace to extract the facial region.
    
    args:
        face_data: dictionary with facial data
        
    return: (x_min, x_max, y_min, y_max) tuple of ints representing facial region for use in numpy slicing
    """
    x_min, y_min, x_max, y_max = face_data['facial_area']
    
    return x_min, x_max, y_min, y_max

def get_mediapipe_faces(image):
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
        return -1
    
    # If multiple faces are found, get the index of the centermost face.
    if len(results.multi_face_landmarks) > 1:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        # Landmarks 475 and 470 are points roughly near the center of each eye.
        # Rough face centers are calculated by taking the average of the x- and y- coordinates of these landmarks.
        rough_face_centers = [np.sum((np.array((face.landmark[475].x, face.landmark[475].y)),
                                      np.array((face.landmark[470].x, face.landmark[470].y))), axis = 0) / 2\
                             for face in results.multi_face_landmarks]
        # MediaPipe uses image normalized coordinates, so they must be converted back to pixel coordinates.
        rough_face_centers_pixel = [normal_to_pixel(f[0], f[1], image.shape[1], image.shape[0]) for f in rough_face_centers]
        results_key = get_index_center_face(image_center, rough_face_centers_pixel)
    
    return results.multi_face_landmarks[results_key]

def mediapipe_align(image, mp_results):
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
            landmark = mp_results.landmark[num]
            coords = normal_to_pixel(float(landmark.x), float(landmark.y), image.shape[1], image.shape[0])
            eye_pixels.append(np.array(coords))
        pixels.append(np.array(eye_pixels))

    # Derive rough eye centers by averaging pixel coordinates from landmarks.
    eye_centers = [(np.sum(eye, axis = 0) / len(eye)).astype(int) for eye in pixels]
    
    angle, direction = get_angle_from_eyes(eye_centers[0], eye_centers[1])

    rotated_image = rotate_from_angle(image, tuple(np.array(image.shape[1::-1]) / 2), angle, direction)
    
    return rotated_image

def get_mediapipe_region(image, mp_results):
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
    x_min = np.clip(0, int(np.round(min([k.x for k in mp_results.landmark]) * image.shape[1])), image.shape[1])
    x_max = np.clip(0, int(np.round(max([k.x for k in mp_results.landmark]) * image.shape[1])), image.shape[1])
    y_min = np.clip(0, int(np.round(min([k.y for k in mp_results.landmark]) * image.shape[0])), image.shape[0])
    y_max = np.clip(0, int(np.round(max([k.y for k in mp_results.landmark]) * image.shape[0])), image.shape[0])

    return x_min, x_max, y_min, y_max

def confirm_valid_list_input(input_, valid_options, sorted_ = False):
    """
    Formats an input into a list and confirms it contains only valid options.
    
    args:
        input_: input to test
        valid_options: list of valid options
        sorted_: if True, return a sorted list, else return an unsorted list
        
    return: list, formatted input; if not valid, returns -1 for further error handling
    """
    # Convert input string to a list for iteration.
    if type(input_) == str:
        input_ = [input_]
    
    # If input type is not a list return -1.
    if type(input_) != list:
        return -1    
    
    # If input list is empty, return -1.
    if input_ == []:
        return -1
    
    # If input type contains invalid values, extract only valid values.
    if list(set(input_).difference(valid_options)) != []:
            input_ = [x for x in input_ if x in valid_options]
            
    if sorted_:
        input_ = sorted(input_)
    
    return input_

def align_image(image, backend = ['dlib', 'retinaface', 'mediapipe']):
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
    
    # Format and confirm backend parameter
    backend = confirm_valid_list_input(backend, valid_options = ['dlib', 'retinaface', 'mediapipe'])
    
    # Handling error code from backend confirmation, exiting due to invalid backend input
    if type(backend) == int:
        return -1
    
    # Loop through backends, when one succeeds, return the rotated image and the backend used.
    for b in backend:
        if b == 'dlib':
            shape_predictor_path = './pretrained_models/shape_predictor_5_face_landmarks.dat'
            face_info = get_dlib_faces(image, shape_predictor_path)
            if face_info == -1:
                continue
            rotated_image = dlib_align(image, face_info)
            
            return rotated_image, b
            
        if b == 'retinaface':
            face_info = get_retinaface_faces(image)
            if face_info == -1:
                continue
            rotated_image = retinaface_align(image, face_info)
            
            return rotated_image, b
        
        if b == 'mediapipe':
            face_info = get_mediapipe_faces(image)
            if face_info == -1:
                continue            
            rotated_image = mediapipe_align(image, face_info)
            
            return rotated_image, b
    
    # If no backends found faces, return -1.
    return -1

def crop_image(image, backend = ['mediapipe', 'retinaface', 'dlib']):
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
    # Format and confirm backend parameter
    backend = confirm_valid_list_input(backend, valid_options = ['mediapipe', 'retinaface', 'dlib'])
    
    # Handling error code from backend confirmation, exiting due to invalid backend input
    if type(backend) == int:
        return -1

    # Loop through backends, when one succeeds, return the cropped image and the backend used.            
    for b in backend:
        if b == 'mediapipe':
            face_info = get_mediapipe_faces(image)
            if face_info == -1:
                continue
            x_min, x_max, y_min, y_max = get_mediapipe_region(image, face_info)
            cropped_image = image[y_min:y_max, x_min:x_max, :]
            
            return cropped_image, b   
            
        if b == 'retinaface':
            face_info = get_retinaface_faces(image)
            if face_info == -1:
                continue            
            x_min, x_max, y_min, y_max = get_retinaface_region(face_info)
            cropped_image = image[y_min:y_max, x_min:x_max, :]
            
            return cropped_image, b   
        
        if b == 'dlib':
            shape_predictor_path = './pretrained_models/shape_predictor_5_face_landmarks.dat'
            face_info = get_dlib_faces(image, shape_predictor_path)
            if face_info == -1:
                continue            
            x_min, x_max, y_min, y_max = get_dlib_region(face_info)
            cropped_image = image[y_min:y_max, x_min:x_max, :]
            
            return cropped_image, b
    
    # If no backends found faces, return -1.   
    return -1

def resize_image(image, end_dim = (152, 152), pad = True, color = (0, 0, 0)):
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
    resized_image = image.copy()
    
    if pad:
        # First, compress image along axes bigger than end dimension using cv2.resize with area interpolation.
        if resized_image.shape[0] > end_dim[0] or resized_image.shape[1] > end_dim[1]:
            resized_image = cv2.resize(resized_image, np.minimum(resized_image.shape[0:-1], end_dim)[::-1],
                                       interpolation = cv2.INTER_AREA)

        # Second, pad image using specified color.
        # We want to pad second so that the interpolation does not pick up the padded data.
        if resized_image.shape[0] < end_dim[0] or resized_image.shape[1] < end_dim[0]:
            old_height, old_width, channels = resized_image.shape
            new_height, new_width = np.maximum(resized_image.shape[0:-1], end_dim)
            x_center = (new_width - old_width) // 2
            y_center = (new_height - old_height) // 2

            y_top = y_center + old_height
            x_right = x_center + old_width

            padded_image = np.full((new_height, new_width, channels), color, dtype = np.uint8)

            padded_image[y_center:y_top, x_center:x_right] = resized_image

            resized_image = padded_image.copy()
    else:
        if resized_image.shape[0] != end_dim[0] or resized_image.shape[1] != end_dim[1]:
            # If purely shrinking, use cv2.INTER_AREA, else, use cv2.INTER_CUBIC
            if resized_image.shape[0] > end_dim[0] and resized_image.shape[1] > end_dim[1]:
                interpolation = cv2.INTER_AREA
            else:
                interpolation = cv2.INTER_CUBIC
            resized_image = cv2.resize(resized_image, end_dim[::-1], interpolation = interpolation)
    
    return resized_image

class ImagePreprocessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.aligned_image = None
        self.cropped_image = None
        self.resized_image = None
        self.vector = None
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
        self.vectorizer = None
        
        return

    def clear_attributes(self):
        self.image = None
        self.aligned_image = None
        self.cropped_image = None
        self.resized_image = None
        self.vector = None
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
        self.vectorizer = None
        
        return
        
    def load_image(self):
        self.image = cv2.imread(self.image_path)
        
    def get_image(self):
        self.load_image()
        return self.image
    
    def get_aligned_image(self):
        return self.aligned_image

    def get_cropped_image(self):
        return self.cropped_image
    
    def get_resized_image(self):
        return self.resized_image
    
    def get_vector(self):
        return self.vector
    
    def set_error_string(self):
        pass
    
    def get_error(self):
        return self.error_code, self.error_string
    
    def preprocess_image(self, alignment_backend = ['dlib', 'retinaface', 'mediapipe'],
                         crop_backend = ['mediapipe', 'retinaface', 'dlib'],
                         end_dim = (152, 152), color = (0, 0, 0), pad = True,
                         steps = ['align', 'crop', 'resize']):
    
        if type(steps) == str:
            if steps == 'full':
                steps = ['align', 'crop', 'resize']
            else:
                steps = [steps]

        steps = confirm_valid_list_input(steps, valid_options = ['align', 'crop', 'resize'])

        if type(steps) == int:
            return -1
        
        self.clear_attributes()
        self.load_image()   
        self.aligned_image = self.image.copy()
        self.cropped_image = self.image.copy()
        self.resized_image = self.image.copy()
        self.resize_dim = end_dim
        self.pad = pad
        self.pad_color = color

        for step in steps:
            if step == 'align':
                self.aligned_image, self.a_backend = align_image(self.image, backend = alignment_backend)
                if type(self.aligned_image) == int:
                    # Checks if alignment backend and crop backend are the same
                    # If all backends fail for alignment, they will fail for backend, so exit
                    if sorted(alignment_backend) == sorted(crop_backend):
                        self.error_code = -1
                        return

                continue

            if step == 'crop':
                self.cropped_image, self.c_backend = crop_image(self.aligned_image, backend = crop_backend)
                if type(self.cropped_image) == int:
                    self.error_code = -1
                    return

                continue

            if step == 'resize':
                self.resized_image = resize_image(self.cropped_image,
                                                  end_dim = self.resize_dim,
                                                  color = self.pad_color,
                                                  pad = self.pad)

                continue

        return 

    def preprocess_ghosh(self):
        self.clear_attributes()
        self.load_image()
        shape_predictor_path = './pretrained_models/shape_predictor_5_face_landmarks.dat'
        self.dlib_details = get_dlib_faces(self.image, shape_predictor_path)
        if type(self.dlib_details) == int:
            self.error_code = -1
            return
        
        self.cropped_image = dlib.extract_image_chip(self.image, self.dlib_details)
        self.resized_image = resize_image(self.cropped_image, end_dim = (152, 152), pad = True, color = (0, 0, 0))

        return

    def vectorize(self):
        self.vector = dp.vectorize_image(self.resized_image, self.vectorizer)

        return

    def set_vectorizer(self, model = None):
        if model is None:
            self.vectorizer = dp.get_deepface_vectorizer('./pretrained_models/VGGFace2_DeepFace_weights_val-0.9034.h5')
        else:
            self.vectorizer = model
            
        return

def preprocess_dataset(image_root = './data/raw', folders = [''], dest_image_root = './data/preprocessed',
                       log_path = '', save_steps = ['align', 'crop', 'resize'],
                       save_folders = ['aligned', 'cropped', 'resized'], preprocess_type = 'normal', 
                       split_results = False, log_failed = False, overwrite = False, **kwargs):
    
    # Parameter Validation
    if folders is None:
        folders = ['']
    folders = confirm_valid_list_input(folders, ['', 'train', 'test'])
    if folders == -1:
        print("Parameter folders must be a string or list of strings. "
              + "Valid strings include 'train', 'test', and ''")
        return
    
    save_steps = confirm_valid_list_input(save_steps, ['align', 'crop', 'resize'], sorted_ = True)
    if save_steps == -1:
        print("Parameter save_steps must be a string or list of strings. "
              + "Valid strings include 'align', 'crop', and 'resize'")
        return
    
    if type(save_folders) == str:
        save_folders = [save_folders]
    if len(save_folders) != len(save_steps):
        print("Must have same number of save_folders as save_steps. "
              + "They must follow the order align_folder, cropped_folder, resized_folder for each save_step included. "
              + f"# of save_steps: {len(save_steps)}, # of save_folders: {len(save_folders)}")
    
    string_params = [image_root, dest_image_root, log_path]

    if not all([type(x) == str for x in string_params]):
        string_param_names = ['image_root', 'dest_image_root', 'log_path']
        indices = np.argwhere(np.where(type(string_params) != str, (True, False)))
        for idx in indices:
            print(f'Parameter {string_param_names[idx]} must be a string. '
                  + f'{string_params[idx]} was input. Please check input data types.')
        return
    
    if preprocess_type != 'normal' and preprocess_type != 'ghosh':
        print("Parameter preprocess_type must have value 'normal' or 'ghosh'.")
        
        return
    
    bool_params = [split_results, log_failed, overwrite]
    
    if not all([type(x) == bool for x in bool_params]):
        bool_param_names = ['split_results', 'log_failed', 'overwrite']
        indices = np.argwhere(np.where(type(bool_params) != bool, (True, False)))
        for idx in indices:
            print(f'Parameter {bool_param_names[idx]} must be a string. '
                  + f'{bool_params[idx]} was input. Please check input data types.')
        return

    bad_files = []
    
    # Loop through folders
    for folder in folders:
        image_names = os.listdir(os.path.join(image_root, folder))

        # For each image get the image path
        for i, image_name in enumerate(image_names):
            print(f'{i + 1} / {len(image_names)}. Processing image {image_name}')
            image_path = os.path.join(image_root, folder, image_name)
            
            # If split, make destination paths based on input folders
            if split_results:
                dest_paths = [os.path.join(dest_image_root, s, folder, image_name) for s in save_folders]
            else:
                dest_paths = [os.path.join(dest_image_root, s, image_name) for s in save_folders]
            
            # If destination folders do not exist, make them
            for dest in dest_paths:
                dest_path_folder = os.path.join(*dest.split(os.path.sep)[0:-1])
                if not os.path.exists(dest_path_folder):
                    os.makedirs(dest_path_folder)
            
            # If overwrite is false, check if image exists at all destination paths
            if not overwrite:
                dest_path_exists = [os.path.exists(p) for p in dest_paths]
                if all(dest_path_exists):
                    print(f'Image already exists at all destination paths.')
                    continue
            
            # Preprocess image
            prep = ImagePreprocessor(image_path)
            
            if preprocess_type == 'normal':
                prep.preprocess_image(**kwargs)
            elif preprocess_type == 'ghosh':
                prep.preprocess_ghosh()
            
            # If an error occurred, add to bad files list and continue
            if prep.error_code == -1:
                print(f'{image_name} was not able to be processed.')
                bad_files.append(image_name)
                
                continue
            
            # Save image to destinations
            for j, step in enumerate(save_steps):
                # If overwrite is false, check if image exists at destination path
                if not overwrite:
                    if dest_path_exists[j]:
                        continue
                if step == 'align':
                    aligned_image = prep.get_aligned_image()
                    print(f'Writing aligned image to {save_folders[j]}')
                    cv2.imwrite(dest_paths[j], aligned_image)
                if step == 'crop':
                    cropped_image = prep.get_cropped_image()
                    print(f'Writing cropped image to {save_folders[j]}')
                    cv2.imwrite(dest_paths[j], cropped_image)
                if step == 'resize':
                    resized_image = prep.get_resized_image()
                    print(f'Writing resized image to {save_folders[j]}')
                    cv2.imwrite(dest_paths[j], resized_image)
    
    # If logging is true, write list of bad files to log path
    if log_failed:
        with open(log_path, 'w') as l:
            for ix, line in enumerate(bad_files):
                if ix != 0:
                    delim = '\n'
                else:
                    delim = ''
                l.write(line + delim)
                
    return