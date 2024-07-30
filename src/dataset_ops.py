import os
import cv2
import pickle
import numpy as np
from . import utils
from .image_preprocessor import ImagePreprocessor
from .image_vectorizer import ImageVectorizer

def preprocess_dataset(image_root = './data/raw', folders = [''], dest_image_root = './data/preprocessed',
                       log_path = '', save_steps = ['align', 'crop', 'resize'],
                       save_folders = ['aligned', 'cropped', 'resized'], preprocess_type = 'normal', 
                       split_results = False, log_failed = False, overwrite = False, **kwargs):
    
    # Parameter Validation
    if folders is None:
        folders = ['']
    folders = utils.confirm_valid_list_input(folders, ['', 'train', 'test'])
    if folders == -1:
        print("Parameter folders must be a string or list of strings. "
              + "Valid strings include 'train', 'test', and ''")
        return
    
    save_steps = utils.confirm_valid_list_input(save_steps, ['align', 'crop', 'resize'], sorted_ = True)
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
                dest_paths = [os.path.join(dest_image_root, s, folder,
                                           image_name.split('.')[0] + '.png') for s in save_folders]
            else:
                dest_paths = [os.path.join(dest_image_root, s,
                                           image_name.split('.')[0] + '.png') for s in save_folders]
            
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
            processor = ImagePreprocessor(image_path)
            
            if preprocess_type == 'normal':
                processor.preprocess_image(**kwargs)
            elif preprocess_type == 'ghosh':
                processor.preprocess_ghosh()
            
            # If an error occurred, add to bad files list and continue
            if processor.error_code == -1:
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
                    aligned_image = processor.get_aligned_image()
                    print(f'Writing aligned image to {save_folders[j]}')
                    cv2.imwrite(dest_paths[j], aligned_image)
                if step == 'crop':
                    cropped_image = processor.get_cropped_image()
                    print(f'Writing cropped image to {save_folders[j]}')
                    cv2.imwrite(dest_paths[j], cropped_image)
                if step == 'resize':
                    resized_image = processor.get_resized_image()
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

def vectorize_processed_dataset(dataset_path, destination_root, path_suffix, vectorizer = None):
    if vectorizer is None:
        # Initialize a vectorizer
        vectorizer = ImageVectorizer('./pretrained_models/VGGFace2_DeepFace_weights_val-0.9034.h5')
        vectorizer.initialize()
    
    # Make sure the destination folder exists
    if not os.path.exists(destination_root):
        os.makedirs(destination_root)
    
    for i, image_name in enumerate(os.listdir(dataset_path)):
        image_path = os.path.join(dataset_path, image_name)
        print(f'Vectorizing image {i + 1} from {image_path}')
        vector_path = os.path.join(destination_root, image_name.split('.')[0] + f'-{path_suffix}.pickle')

        # If the vector already exists (possibly from a prior session), skip
        if os.path.exists(vector_path):
            print(f'Vector {i + 1} already exists at {vector_path}')
            continue

        # Get vector and write to disk
        vector = vectorizer.vectorize_image(image_path, preprocess = False)

        with open(vector_path, 'wb') as f:
            print(f'Pickling vector {i + 1} at {vector_path}')
            pickle.dump(vector, f)

    return