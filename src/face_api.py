from . import deep_vectorizer as dp
from . import utils
from .image_preprocessor import ImagePreprocessor

def init_model(weights_path = './pretrained_models/VGGFace2_DeepFace_weights_val-0.9034.h5'):
    model = dp.get_deepface_vectorizer(weights_path)
    return model
    
def init_vectors():
    # Calls Database API function get all vectors from database into memory
#     return get_all_vectors()
    pass
    
def get_image_vector(image_path, model = None):
    image_prep = ImagePreprocessor(image_path)
    image_prep.preprocess_image()
    image_prep.vectorize(model)

    vector = image_prep.get_vector()

    return vector

def find_image_match(image_path, db_vectors, model = None, metric = 'l2', threshold_criterion = None):
    # Select a number of threshold string options for picking a threshold criterion
    # To be determined after analysis
    
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
            print("Invalid metric. Choose 'l2' or 'cos'.")
            return
        if distance < match_results:
            match_results = distance
            match_id = key
    
    if match_results < threshold:
        return match_id, match_results, threshold, metric
    else:
        print(f"No matching images found.\nDistance threshold: {threshold}\nLowest Distance: {match_results}")