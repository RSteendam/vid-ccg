import pickle
import numpy as np

def load_inverse_permutation(permatutation_maps_filename, model_key):
    with open(permatutation_maps_filename, "rb") as f:
        permutation_maps = pickle.load(f)
    
    permutation = permutation_maps[model_key]
    
    inverse_permutation = get_inverse_permutation(permutation)
    
    return inverse_permutation

def get_inverse_permutation(perm: np.ndarray):
    inverse_perm = np.arange(perm.shape[0])
    inverse_perm[perm] = inverse_perm.copy()
    
    return inverse_perm