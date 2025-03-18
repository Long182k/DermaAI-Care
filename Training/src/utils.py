import os
import random
import numpy as np
import tensorflow as tf

def set_global_seeds(seed=42):
    """
    Set global random seeds for reproducibility across multiple libraries
    
    Args:
        seed: Integer seed value to use
    """
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set TensorFlow's random seed
    tf.random.set_seed(seed)
    
    # Set Python's hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set TensorFlow's deterministic operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    print(f"Global random seeds set to {seed} for reproducibility")