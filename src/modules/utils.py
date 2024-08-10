import os
import pickle
import json
import numpy as np
from numpy.linalg import norm
import pandas as pd

def l2_distance(vector_1, vector_2):
    return norm(vector_1 - vector_2, ord = 2)

def cosine_distance(vector_1, vector_2):
    return 1 - (vector_1 @ vector_2.T)/(norm(vector_1)*norm(vector_2))

# Sum of first n natural numbers for verifying intra-label distance counts per label
def sum_first_n(n):
    return (n * (n + 1)) / 2

# Returns sorted list of all unique names of files in path
def get_unique_names(path, string_delim = '_', slice_tuple = (0, -2)):
    return sorted(list(set([string_delim.join(x.split(string_delim)[slice_tuple[0]:slice_tuple[1]])
                            for x in os.listdir(path)])))

def get_all_paths(root_path):
    all_paths = sorted([x for x in os.listdir(root_path)])
    return all_paths

# Helper function to load an object from a pickled file
def unpickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

# Load a dictionary from a json file
def load_json(read_path):
    with open(read_path, 'r') as f:
        result = json.load(f)
    return result

def load_json_into_frame(json_path):
    # Read intra-label distances from disk and place into dataframe
    distances = load_json(json_path)
    distances_df = pd.DataFrame.from_dict(distances, orient = 'index').reset_index()
    distances_df.columns = ['name', 'l2_distances', 'cosine_distances']
    
    # Remove rows with no distances
    distances_df = distances_df.loc[distances_df.l2_distances.apply(len).gt(0)]
    
    return distances_df

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