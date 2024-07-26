import numpy as np
from numpy.linalg import norm

def l2_distance(vector_1, vector_2):
    return norm(vector_1 - vector_2, ord = 2)

def cosine_distance(vector_1, vector_2):
    return 1 - (vector_1 @ vector_2.T)/(norm(vector_1)*norm(vector_2))