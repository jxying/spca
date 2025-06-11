import numpy as np
from src.spca import spca

def test_spca_runs_and_returns_reasonable_distance():
    # Set a small problem for fast test
    p = 10
    k = 2
    n = 500
    max_iter = 10
    sparsity_level = k

    # Construct a sparse vector
    true_vector = np.zeros(p)
    support = [1, 7]
    true_vector[support] = [1, -1]
    true_vector /= np.linalg.norm(true_vector, 2)

    # Simulate data
    signal_strength = 10
    latent_factors = np.random.randn(n)
    noise = np.random.multivariate_normal(np.zeros(p), np.eye(p), size=n).T
    data_matrix = np.sqrt(signal_strength) * (true_vector[:, None] @ latent_factors[None, :]) + noise

    distance, elapsed = spca(true_vector, data_matrix, p, k, n, sparsity_level, max_iter)

    # Test output type and range
    assert isinstance(distance, float)
    assert 0 <= distance < 2, "Distance should be between 0 and 2 (since both are unit vectors up to sign)"
    assert isinstance(elapsed, float)

def test_spca_converges_fast_on_easy_case():
    # Easy case with k=1 (single nonzero)
    p = 5
    k = 1
    n = 50
    max_iter = 10
    sparsity_level = k

    true_vector = np.zeros(p)
    true_vector[2] = 1

    signal_strength = 10
    latent_factors = np.random.randn(n)
    noise = np.random.multivariate_normal(np.zeros(p), np.eye(p), size=n).T
    data_matrix = np.sqrt(signal_strength) * (true_vector[:, None] @ latent_factors[None, :]) + noise

    distance, _ = spca(true_vector, data_matrix, p, k, n, sparsity_level, max_iter)
    assert distance < 0.5, f"Distance too large: {distance}"
