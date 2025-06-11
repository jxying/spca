import numpy as np
from src.truncation import truncation
import time

def spca(true_vector, data_matrix, p, k, n, sparsity_level, max_iter):
    """
    Sparse PCA using truncated power iteration.

    Args:
        true_vector (np.ndarray): Ground truth signal vector, shape (p,).
        data_matrix (np.ndarray): Observed data matrix, shape (p, n).
        p (int): Dimension of the signal.
        k (int): Sparsity of the signal.
        n (int): Number of samples.
        sparsity_level (int): Sparsity in the truncated power iteration.
        max_iter (int): Maximum number of power iterations.

    Returns:
        min_distance (float): Distance between true_vector and estimated vector.
        elapsed (float): Elapsed time in seconds.
    """
    start_time = time.time()
    # Compute sample covariance matrix
    sample_cov = data_matrix @ data_matrix.T / n  # shape (p, p)
    sample_cov_diag = np.diag(sample_cov)
    # Find the index with the largest diagonal entry
    max_diag_index = np.argmax(sample_cov_diag)
    # Select support indices corresponding to the largest k absolute values in the max_diag_index column
    support_indices = np.argsort(np.abs(sample_cov[:, max_diag_index]))[-k-1:-1]
    support_indices = np.sort(support_indices)
    
    # Extract the submatrix and compute the leading eigenvector
    submatrix = sample_cov[np.ix_(support_indices, support_indices)]
    eigvals, eigvecs = np.linalg.eigh(submatrix)
    initial_vector = np.zeros(p)
    initial_vector[support_indices] = eigvecs[:, -1]  # leading eigenvector

    # Truncated power iteration
    for _ in range(max_iter):
        prev_vector = initial_vector.copy()
        initial_vector = sample_cov @ prev_vector
        initial_vector = truncation(initial_vector, sparsity_level)
        initial_vector = initial_vector / np.linalg.norm(initial_vector, 2)
        # Check for convergence
        if np.linalg.norm(initial_vector - prev_vector, 2) < 1e-8:
            break

    elapsed = time.time() - start_time
    # Return the minimum distance (accounting for sign ambiguity)
    min_distance = min(
        np.linalg.norm(true_vector - initial_vector, 2),
        np.linalg.norm(true_vector + initial_vector, 2)
    )
    return min_distance, elapsed
