# Project Overview

This project is designed to use Facebook's DeepFace neural net to create a proof-of-concept facial recognition architecture. Created as part of the final for CS661 Python Programming at Pace University. For a full description of the project, including references to all sources, read the [paper](./paper/CS661-Python_Programming-Final_Paper.pdf).

The facial recognition engine uses the pre-trained weights developed by [Swarup Ghosh](https://github.com/swghosh/DeepFace/releases) using the VGG-Face2 dataset. By default, the facial recognition uses Ghosh's preprocessing pipeline, as the facial recognizer works best when using the preprocessing pipeline on which it was trained. However, additional preprocessing backends are available using MediaPipe and RetinaFace, as documented under ./src/image_preprocessor.py. 

The main demonstration of this project's capabilities are through the interactive Jupyter Notebook Demo-and-Test.ipynb located in the root of this repository. A version using Kivy for the UI and Firebase Cloud Firestore for the database is accessible through the application_ui.py and seed_vectors.py files located under ./src off of the feat/seed branch, although the functionality is not completely integrated yet with the facial recognition engine.

# Setup

## Libraries
Use pip install to install the following packages. Alternatively, put the following command into a terminal window.
<p>
  <ul>
    <li>pip install -r requirements.txt</li>
  </ul>
</p>

Note that there may by dependency conflicts when trying to use the Kivy implementation. This is a known bug.
  
```
# Libraries needed for Jupyter Notebooks and Facial Recognition Engine
dlib==19.24.4
mediapipe==0.10.11
numpy==1.23.0
opencv_python==4.8.0.76
pandas==1.5.2
retina_face==0.0.17
tensorflow==2.13.1

# Libraries needed for Kivy-based UI and Firebase DB
firebase_admin==6.5.0
Kivy==2.3.0
opencv_contrib_python==4.10.0.84
```

## Python and Jupyter Notebooks Versions

The Jupyter Notebook demo and all notebooks in the repository were successfully run under Python 3.8.10.
The Kivy UI application was tested under Python 3.9.
The Jupyter Notebook Demo was successfully tested using the following versions:

```
IPython          : 8.7.0
ipykernel        : 6.19.4
ipywidgets       : 8.0.4
jupyter_client   : 7.4.8
jupyter_core     : 5.1.0
jupyter_server   : 2.0.4
jupyterlab       : not installed
nbclient         : 0.7.2
nbconvert        : 7.2.7
nbformat         : 5.7.1
notebook         : 6.5.2
qtconsole        : 5.4.0
traitlets        : 5.8.0
```

## Directory Structure Required for Jupyter Notebook Demo-and-Test.ipynb
  
Note that dummy_dataset and personal_images_batch2 datasets are pre-built sets for demonstration purposes. Any folder name can be placed into ./data for convenience of access.

For preprocessing of custom data, consider using the Preprocessing notebook in ./src (not shown below) or by using the functions in ./src/modules/dataset_ops.py, image_preprocessor.py and image_vectorizer.py.

Pre-trained models must be obtained from the Github sources specified in the next section, unpacked and placed in ./pretrained_models.


```
.
|── Demo-and-Test.ipynb
|── data/
|   ├── dummy_dataset/
|   |	├──input_images/
|   |	├── reference_images/
|   |	└── reference_vectors/
|   └── personal_images_batch2/
|	├──input_images/
|	├── reference_images/
|	└── reference_vectors/
|── pretrained_models/
|   ├── shape_predictor_5_face_landmarks.dat
|   └── VGGFace2_DeepFace_weights_val-0.9034.h5
|── src/
|   └── modules/
|	├── data_manipulation.py
|	├── dataset_ops.py
|	├── face_api.py
|	├── face_utils.py
|	├── image_preprocessor.py
|	├── image_vectorizer.py
|	├── __init__.py
|	├── stat_utils.py
|	└── utils.py
```
</details>

## External Pre-trained Models Needed
The shape_predictor_5_face_landmarks.dat.bz2 file from [davisking's Github](https://github.com/davisking/dlib-models) is used by the dlib library for detecting faces and must be unpacked and placed into ./pretrained_models. The VGGFace2_DeepFace_weights_val-0.9034.h5.zip from [Swarup Ghosh's Github](https://github.com/swghosh/DeepFace/releases) is the file containing the model weights used by the DeepFace neural net and must be unzipped and placed into ./pretrained_models.

Example unpacking Linux commands using a terminal (from inside the ./pretrained_models directory):

<p>
  <ul>
    <li>bzip2 -d shape_predictor_5_face_landmarks.dat.bz2</li>
    <li>unzip VGGFace2_DeepFace_weights_val-0.9034.h5.zip</li>
  </ul>
</p>

# How to Use Demo and Test Jupyter Notebook
The Demo and Test notebook is set to be run with very few required inputs from the user.

Cell 3 is where the user specifies the threshold strategy, threshold strictness, and maximum number of matches they wish for the facial recognizer to return. For definitions of threshold strictness and threshold strategy, see ./src/modules/face_api.py. The balanced_acc strategy with strictness 5 is the most lenient strategy and is the best set of parameters for manually inspecting the effectiveness of the facial recognition engine by allowing a wider variety of matches.

Cells 4 and 5 are where the user specifies the directories containing the reference images, reference vectors, and input images they wish to use. The number of digits should be specified and is an indicator of how many digits post-pend the image name. For example, "Wayne_Gretsky_0002.jpg" has 4 digits (0002) while "Dan1-02.jpg" has 2 digits (02). All vectors in the reference vector directory must follow the same digit schema.

Cells 6 and 7 display the base image for inspection of the image prior to matching.

Cell 8 is where find_image_match() is called, and cell 9 automatically visualizes the results. Note that cell 9 shows the input image and all matched images in one row, and for large match_num parameters, this may result in small images and poor visualization. A maximum match_num of 5 is recommended for this reason. If the user wishes to test more images or directories, they can simply change the parameters in cells 3, 4 and 5, and rerun the rest of the notebook to retrieve new results.
