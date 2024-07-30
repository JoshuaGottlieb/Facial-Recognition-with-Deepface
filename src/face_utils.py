import math
import cv2
import numpy

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