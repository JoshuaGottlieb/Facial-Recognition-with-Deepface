from . import utils
from .image_preprocessor import ImagePreprocessor
from .image_vectorizer import ImageVectorizer

def init_model(weights_path = './pretrained_models/VGGFace2_DeepFace_weights_val-0.9034.h5'):
    """
    Initializes a vectorizer for processing images.
    
    args:
        weights_path: str representing path to pre-trained model weights (.h5 file)
        
    returns: An initialized tensorflow.keras.Model object
    """
    model = ImageVectorizer(weights_path)
    model.initialize()
    return model
    
def init_vectors():
    # Calls Database API function get all vectors from database into memory
#     return get_all_vectors()
    pass
    
def get_image_vector(image_path, model = None, preprocess_type = 'normal', **kwargs):
    """
    Generates a vector from an image loaded from an image path.
    
    args:
        image_path: str representing path to image to be processed
        model: tensorflow.keras.Model object to use to generate predictions.
               If no model given, one will be initialized; however, work is much faster using a pre-initialized model.
        preprocess_type: str representing the type of preprocessing to perform. Valid parameters are 'normal' and 'ghosh'
        **kwargs: key-word arguments to be passed to the image preprocessor
    """
    if model is None:
        model = init_model()
    
    if preprocess_type != 'normal' or preprocess_type != 'ghosh':
        print(f'Invalid preprocess type specified, defaulting to normal preprocessing with given key-word arguments.')
        preprocess_type = 'normal'
    
    vector = model.vectorize_image(image_path, preprocess_type = preprocess_type, **kwargs)

    return vector

def find_image_match(image_path, db_vectors, model = None, metric = 'l2', threshold_criterion = None):
    """
    Finds the best matching image to an image loaded from an image path in a database by converting the image to a vector. 
    Compares the vector to database vectors and returns the database id of the closest vector and matching information.
    
    args:
        image_path: str representing the image to be processed
        db_vectors: dict of {id:vector} key-value pairs to compare against
        model: tensorflow.keras.Model object to use to generate predictions.
               If no model given, one will be initialized; however, work is much faster using a pre-initialized model.
        metric: str representing the distance metric to use. Valid parameters are 'l2' and 'cos'
        threshold_criterion: str representing the criterion to maximize when selecting the threshold.
                             Criterions and thresholds to be decided through analysis.
    
    """
    # Select a number of threshold string options for picking a threshold criterion; to be determined after analysis
    # Logic involving setting thresholds based on criterion here
    
    vector = get_image_vector(image_path, model = model)
    
    match_id = ''
    match_results = 1000
    
    # Using structure from database API function - say a dictionary
    for key, value in db_vectors.items():
        if metric == 'l2':
            # dummy placeholder threshold
            threshold = 100
            distance = utils.l2_distance(vector, value)
        elif metric == 'cos':
            # dummy placeholder threshold
            threshold = 100
            distance = utils.cosine_distance(vector, value)
        else:
            print("Invalid metric. Choose 'l2' or 'cos'")
            return
        if distance < match_results:
            match_results = distance
            match_id = key
    
    if match_results < threshold:
        return match_id, match_results, threshold, metric
    else:
        print(f"No matching images found.\nDistance threshold: {threshold}\nLowest Distance: {match_results}")
        return