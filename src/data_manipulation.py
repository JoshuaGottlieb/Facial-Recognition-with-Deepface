import os
import pickle
import json
from . import calculation
import numpy as np
import pandas as pd
import re

# Returns sorted list of all unique names of files in path
def get_unique_names(path, string_delim = '_', slice_tuple = (0, -2)):
    return sorted(list(set([string_delim.join(x.split(string_delim)[slice_tuple[0]:slice_tuple[1]])
                            for x in os.listdir(path)])))

# Helper function to load a vector from a pickled object
def load_vector(path):
    with open(path, 'rb') as f:
        vector = pickle.load(f)
    return vector


# Load a dictionary from a json file
def load_json(read_path):
    with open(read_path, 'r') as f:
        result = json.load(f)
    return result


def get_all_paths(root_path):
    all_paths = sorted([x for x in os.listdir(root_path)])
    return all_paths

# Get all path leaves and load all vectors from path root
def get_all_paths_and_vectors(root_path):
    all_paths = get_all_paths(root_path)
    vectors = [load_vector(os.path.join(root_path, x)) for x in all_paths]

    return all_paths, vectors

# Sum of first n natural numbers for verifying intra-label distance counts per label
def sum_first_n(n):
    return (n * (n + 1)) / 2

# Calculates intra-label distances and writes to disk as a json file
def calculate_intra_distances(unique_names, all_paths, vectors, write_path,
                              string_delim = '_', slice_tuple = (0, -2)):
    # Calculating intra-label distances and writing to a dictionary
    intra_distances = {}

    for name in unique_names:
        intra_distances[name] = {'l2_distances': [], 'cosine_distances': []}
        print(f'Calculating intra-label distances for {name}')
        
        # Grab only vectors with exact name, if only one name is found, no intra-label distances to calculate
        vector1_indices = sorted([all_paths.index(x) for x in all_paths
                                  if string_delim.join(x.split(string_delim)[slice_tuple[0]:slice_tuple[1]]) == name])
        if len(vector1_indices) == 1:
            print(f'Only 1 image for {name}, no intra-label distance exists.')
            continue

        name_l2_distances = []
        name_cos_distances = []

        # If multiple vectors found, compare forward with each other vector under the same label
        # Note that d(a,b) = d(b, a) so we only need to compare forwards
        for i, index_1 in enumerate(vector1_indices):
            # If at last vector in list, no further distances to calculate
            if i == len(vector1_indices) - 1:
                continue
            vector2_indices = sorted(vector1_indices[i + 1:])

            # Calculate distances
            for index_2 in vector2_indices:
                name_l2_distances.append((f'{all_paths[index_1]}', f'{all_paths[index_2]}',
                                          calculation.l2_distance(vectors[index_1],
                                                                  vectors[index_2]).astype(float)))
                name_cos_distances.append((f'{all_paths[index_1]}', f'{all_paths[index_2]}',
                                           calculation.cosine_distance(vectors[index_1],
                                                                       vectors[index_2]).astype(float)))
        # Append distances to dictionary
        print(f'Appending inter-label distances for {name} to dictionary')
        intra_distances[name]['l2_distances'] = name_l2_distances
        intra_distances[name]['cosine_distances'] = name_cos_distances
        
    # Write intra-label distances to disk
    with open(write_path, 'w') as f:
        json.dump(intra_distances, f, indent = '\t')
        
    return

# Calculates inter-label distances and writes to disk as json files
# Note that because of the sheer number of inter-label distances, batching is needed to not run out of memory
def calculate_inter_distances(unique_names, all_paths, vectors, write_path_stem, batch_size = 500,
                              string_delim = '_', slice_tuple = (0, -2)):
    inter_distances = {}

    for i, name in enumerate(unique_names):
        inter_distances[name] = {'l2_distances': [], 'cosine_distances': []}
        print(f'Calculating inter-label distances for {name}')
        
        # Grab vectors with exact name
        vector1_indices = sorted([all_paths.index(x) for x in all_paths
                                  if string_delim.join(x.split(string_delim)[slice_tuple[0]:slice_tuple[1]]) == name])
        
        # Grab comparison vectors with different names
        vector2_indices = sorted([all_paths.index(x) for x in all_paths
                                  if string_delim.join(x.split(string_delim)[slice_tuple[0]:slice_tuple[1]]) != name])
        
        # Take advantage of the fact that the vectors are sorted to choose only indices past the current name
        # Since d(a,b) = d(b,a), we only need to compare forwards, which is important given the huge number of distances
        vector2_indices = sorted([x for x in vector2_indices if x > max(vector1_indices)])

        name_l2_distances = []
        name_cos_distances = []

        # Calculate distances
        for index_1 in vector1_indices:
            for index_2 in vector2_indices:
                name_l2_distances.append((f'{all_paths[index_1]}', f'{all_paths[index_2]}',
                                          calculation.l2_distance(vectors[index_1],
                                                                  vectors[index_2]).astype(float)))
                name_cos_distances.append((f'{all_paths[index_1]}', f'{all_paths[index_2]}',
                                           calculation.cosine_distance(vectors[index_1],
                                                                       vectors[index_2]).astype(float)))
        # Append distances to dictionary
        print(f'Appending inter-label distances for {name} to dictionary')
        inter_distances[name]['l2_distances'] = name_l2_distances
        inter_distances[name]['cosine_distances'] = name_cos_distances

        # Every batch_size names, save dictionary to disk and clear the dictionary to free used memory
        if i != 0 and (i % batch_size == 0 or i == len(unique_names) - 1):
            json_path = f'{write_path_stem}{i:04d}.json'
            print(f'Writing inter-distances intermediate results to {json_path}')
            with open(json_path, 'w') as f:
                json.dump(inter_distances, f, indent = '\t')

            inter_distances = {}
            
    return

def load_json_into_frame(json_path):
    # Read intra-label distances from disk and place into dataframe
    distances = load_json(json_path)
    distances_df = pd.DataFrame.from_dict(distances, orient = 'index').reset_index()
    distances_df.columns = ['name', 'l2_distances', 'cosine_distances']
    
    # Remove rows with no distances
    distances_df = distances_df.loc[distances_df.l2_distances.apply(len).gt(0)]
    
    return distances_df

def confirm_intra_distance_counts(intra_df, all_paths, string_delim = '_', slice_tuple = (0, -2)):
    # Verifying correct counts for each label
    intra_df_counts = intra_df[['name', 'l2_distances']].explode('l2_distances').groupby('name').count().reset_index()
    intra_df_counts.columns = ['name', 'counts']
    intra_df_counts['correct_counts'] = intra_df_counts.name.apply(lambda x:\
                                       sum_first_n(len([y for y in all_paths
                                            if string_delim.join(y.split(string_delim)[slice_tuple[0]:slice_tuple[1]])
                                                                    == x]) - 1))
    intra_df_counts['proper_count'] = np.where(intra_df_counts.counts == intra_df_counts.correct_counts, True, False)
    intra_df_counts.loc[intra_df_counts.proper_count == False]
    
    return intra_df_counts

def get_distance_df(df, distance_string):
    distance_df = df[[distance_string]]
    distance_df = distance_df.explode(distance_string)
    distance_df[distance_string] = distance_df[distance_string].apply(lambda x: x[-1])
    
    return distance_df

def write_inter_batch_frames(json_paths, dest_path_stem):
    # For each json batch, load into a dataframe and process
    # Dataframes become extremely large, especially when applying explode operations
    # This processing was able to be done with 20 GB of memory
    for js in json_paths:
        js_batch_name = int(re.search(r'distances(\d+)\.', js.split('_')[-1])[1])

        # Load json into dictionary
        with open(js, 'r') as f:
            temp_dict = json.load(f)

        # Create dataframe using full dictionary, remove empty entries (there should be none)
        temp_df = pd.DataFrame.from_dict(temp_dict, orient = 'index').reset_index()
        temp_df = temp_df.loc[temp_df.l2_distances.apply(len).gt(0)]
        # Unallocate the dictionary to free memory
        temp_dict = 0
        # Explode around l2 distances and extract image pairs
        temp_df_l2 = temp_df[['l2_distances']].explode('l2_distances')
        temp_df_l2['image_1'] = temp_df_l2.l2_distances.apply(lambda x: '_'.join(x[0].split('_')[0:-1]))
        temp_df_l2['image_2'] = temp_df_l2.l2_distances.apply(lambda x: '_'.join(x[1].split('_')[0:-1]))
        temp_df_pairs = temp_df_l2.drop(['l2_distances'], axis = 1)
        # Write pair information for batch to disk and unallocate the dataframe to free memory
        temp_df_pairs.to_csv(f'{dest_path_stem}_{js_batch_name:04d}_pairs.csv', index = False)
        temp_df_pairs = 0
        # Write l2 distance information for batch to disk and unallocate the dataframe to free memory
        temp_df_l2 = temp_df_l2.drop(['image_1', 'image_2'], axis = 1)
        temp_df_l2.l2_distances = temp_df_l2.l2_distances.apply(lambda x: x[-1])
        temp_df_l2.to_csv(f'{dest_path_stem}_{js_batch_name:04d}_l2.csv', index = False)
        temp_df_l2 = 0
        # Explode around cosine distances and unallocate original dataframe.
        temp_df_cos = temp_df[['cosine_distances']].explode('cosine_distances')
        temp_df = 0
        # Write cosine distance information for batch to disk and unallocate the dataframe to free memory
        temp_df_cos.cosine_distances = temp_df_cos.cosine_distances.apply(lambda x: x[-1])
        temp_df_cos.to_csv(f'{dest_path_stem}_{js_batch_name:04d}_cos.csv', index = False)
        temp_df_cos = 0

    return

def get_batch_paths(path_stem, ext, root = './data/vectorized', other_identifier = ''):
    return [os.path.join(root, x) for x in os.listdir(root)
            if all([path_stem in os.path.join(root, x), ext in x, other_identifier in x])]

def get_unbatched_frame(path_stem, table_identifier, root = './data/vectorized'):
    # Load all inter-label batch csvs into dataframes
    csvs = get_batch_paths(path_stem, '.csv', root = root, other_identifier = table_identifier)
    dfs = [pd.read_csv(c) for c in csvs]
    
    # Concatenate into single dataframe
    unbatched_df = pd.concat(dfs)
    
    return unbatched_df

def confirm_inter_distance_counts(path_stem, vector_leaf):
    # Load all inter-label pair information csvs into a single concatenated dataframe
    pair_df = get_unbatched_frame(path_stem, 'pairs')
    # Convert the dataframe from two columns representing pairs to single column representing image names
    pair_df_stacked = pd.concat([pair_df[['image_1']], pair_df[['image_2']]])
    pair_df_stacked.image_1 = pair_df_stacked.image_1.fillna(pair_df_stacked.image_2)
    pair_df_stacked = pair_df_stacked.drop(['image_2'], axis = 1)
    pair_df_stacked.columns = ['image']
    pair_df_stacked['counts'] = 0
    # Perform grouby operation to check the appropriate number of instances exist for each image
    pair_df_counts = pair_df_stacked.groupby('image').count().reset_index()
    # Collect all image_names
    image_names = pair_df_counts.image.values.tolist()
    slice_index = len(vector_leaf) + 5
    # Calculate the number of instances that should exist for each image
    # Number of images - number of images with same name
    pair_df_counts['correct_counts'] = pair_df_counts.image.apply(lambda x: len(image_names)
                                                                  - len([y for y in image_names
                                                                         if y[:-slice_index] == x[:-slice_index]]))
    pair_df_counts['proper_count'] = np.where(pair_df_counts.counts == pair_df_counts.correct_counts, True, False)
    
    return pair_df_counts