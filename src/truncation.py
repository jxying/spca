import numpy as np

def truncation(vector, s):
    """
    Keep the top s elements of the input vector (by absolute value), set others to zero.
    
    Args:
        vector (np.ndarray): Input 1D array (vector).
        s (int): Number of largest-magnitude elements to keep.
        
    Returns:
        np.ndarray: Truncated vector (same shape as input).
    """
    num_zeros = len(vector) - s
    truncated_vector = np.copy(vector)
    # Find the indices of the smallest (len-s) absolute values
    indices_to_zero = np.argsort(np.abs(vector))[:num_zeros]
    truncated_vector[indices_to_zero] = 0
    return truncated_vector
    
    
