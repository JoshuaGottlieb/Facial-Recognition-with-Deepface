import os
import re
import numpy as np
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
    
def load_vector_dict(dir_path):
    """
    Loads a dictionary with keys on the image_name and values of vectors. Used for testing on local machine.
    
    args:
        dir_path: str representing path to directory with reference vectors.
    """
    vector_leaves = utils.get_all_paths(dir_path)
    vector_paths = [os.path.join(dir_path, leaf) for leaf in vector_leaves]
    
    vector_dict = {re.search(r'([\w_\-]+\d{4})[\w_\-]*\.pickle$', p)[1] : utils.unpickle(p) for p in vector_paths}
      
    return vector_dict
    
def get_image_vector(image_path, model = None, preprocess_type = 'ghosh', **kwargs):
    """
    Generates a vector from an image loaded from an image path.
    
    args:
        image_path: str representing path to image to be processed
        model: tensorflow.keras.Model object to use to generate predictions.
               If no model given, one will be initialized; however, work is much faster using a pre-initialized model.
        preprocess_type: str representing the type of preprocessing to perform. Valid parameters are 'normal' and 'ghosh'
        **kwargs: key-word arguments to be passed to the image preprocessor if preprocess_type is set to 'normal'
        
    returns: np.array representing the image vector
    """
    if model is None:
        model = init_model()
    
    if preprocess_type != 'normal' and preprocess_type != 'ghosh':
        print(f'Invalid preprocess type specified, defaulting to Ghosh preprocessing.')
        preprocess_type = 'ghosh'
    
    vector = model.vectorize_image(image_path, preprocess_type = preprocess_type, **kwargs)

    return vector

def find_image_match(image_path, db_vectors, model = None, metric = 'cos',
                     threshold_strategy = 'matthews_cc', threshold_strictness = 2, match_num = 1):
    """
    Finds the best matching image to an image loaded from an image path in a database by converting the image to a vector. 
    Compares the vector to database vectors and returns the database id of the closest vector and matching information.
    
    args:
        image_path: str representing the image to be processed
        db_vectors: dict of {id:vector} key-value pairs to compare against
        model: tensorflow.keras.Model object to use to generate predictions.
               If no model given, one will be initialized; however, work is much faster using a pre-initialized model.
        metric: str representing the distance metric to use. Valid parameters are 'cos' and 'l2' (not implemented).
        threshold_strategy: str representing the criterion to maximize when selecting the threshold. Criterion listed below.
            'matthews_cc': Matthews Correlation Coefficient. Attempts to balance predictive recall on both classes with
                           predictive precision on both classes. Default criterion. Threshold = 0.269697
            'f1_score': F1-Measure. Attempts to balance precision and recall on the positive class (images match).
                        Less precise than matthew_cc, with slightly better recall. Threshold = 0.293939
            'balanced_acc': Balanced Accuracy. Maximizes accuracy across class imbalances. Prediction favors matching the
                            underlying distribution of the data, which tends to favor the negative class (images do not
                            match) and has low precision, which may result in more false positives. Threshold = 0.447475
            'precision': Precision; positive predictive value. Emphasizes that predictions on the positive class (images
                         match) should come from the positive class, reducing the number of false positives. High
                         precision naturally reduces the ability to identify any positive predictions, which may lead to
                         low identification rate for matching images; i.e., low recall. Threshold = 0.253535
        threshold_strictness: int representing the number of successively less strict thresholds to test.
                              Valid options range from 1 to 5. Used to allow looser matches along similar threshold criterion.
        match_num: int representing the top N matches to return. Default 1.
    
    returns:
        matches: list of match tuples of form ('match_id', 'distance')
        threshold: float representing highest threshold allowed for matching distances
        metric: string representing metric used for distances
    """
    
    # Set thresholds based on selected strategy
    if threshold_strategy == 'matthews_cc':
        thresholds = [0.269697, 0.277778, 0.285859, 0.293939, 0.302020]
    elif threshold_strategy == 'f1_score':
        thresholds = [0.293939, 0.310101, 0.318182, 0.326263, 0.334343]
    elif threshold_strategy == 'balanced_acc':
        thresholds = [0.447475, 0.455556, 0.463636, 0.471717, 0.479798]
    elif threshold_strategy == 'precision':
        thresholds = [0.253535, 0.261616, 0.269697, 0.277778, 0.285859]
    else:
        print("threshold_strategy should have a value among 'matthews_cc', 'f1_score', 'balanced_acc', or 'precision'\n"
              + "Defaulting to 'matthews_cc' threshold_strategy.")
        thresholds = [0.269697, 0.277778, 0.285859, 0.293939, 0.302020]
    
    # Check threshold strictness to determine max number of thresholds to test against
    if not 1 <= threshold_strictness <= 5:
        print("threshold_strictness should have an integer value between [1, 5]. Defaulting to 2.")
        threshold_strictness = 2
    
    # Check number of matches to return
    if match_num <= 0:
        match_num = 1
    
    vector = get_image_vector(image_path, model = model)
    
    # Initialized list to capture ids and distances
    ids = []
    distances = []
    
    # Using dictionary of reference vectors from database API function
    for key, value in db_vectors.items():
        if metric == 'cos':
            distance = utils.cosine_distance(vector, value)
        else:
            print("Invalid metric. Defaulting to cosine distances.")
            metric = 'cos'
            distance = utils.cosine_distance(vector, value)
#         elif metric == 'l2':
#             distance = utils.l2_distance(vector, value)
        
        ids.append(key)
        distances.append(distance)
    
    ids = np.array(ids)
    distances = np.array(distances)
    # Select match_num indices from the match_num smallest distances
    possible_indices = np.argpartition(distances, match_num)[:match_num]
    
    # Check against thresholds for matches
    for i in range(0, threshold_strictness):
        match_indices = np.nonzero(distances[possible_indices] < thresholds[i])
        if len(match_indices[0]) != 0:
            break
    
    # If no matches found below threshold, exit with comparison evaluations
    if len(match_indices[0]) == 0:
        print(f"No matching images found.\nDistance threshold: {thresholds[threshold_strictness - 1]}"
              + f"\nLowest Distance: {min(distances[possible_indices])}")
        return
    
    # Otherwise, return all matches (up to match_num) found, sorted by lowest distance
    match_ids = ids[possible_indices[match_indices[0]]]
    match_distances = distances[possible_indices[match_indices[0]]]
    return sorted(tuple(zip(match_ids, match_distances)), key = lambda x: x[1]), thresholds[threshold_strictness - 1], metric