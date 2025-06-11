import numpy as np
from src.spca import spca

np.random.seed(100)

signal_strength = 10
max_iter = 1000

p = 1000
k = int(round(0.01 * p))
n = int(round(1000 * k * np.log(p)))

# Generate sparse unit-norm signal
true_vector = np.zeros(p)
support = np.random.permutation(p)[:k]
support_values = np.random.randn(k)
support_values[support_values >= 0] = 1
support_values[support_values < 0] = -1
true_vector[support] = support_values / np.linalg.norm(support_values, 2)
sparsity_level = k

# Generate data matrix
latent_factors = np.random.randn(n)
noise = np.random.multivariate_normal(mean=np.zeros(p), cov=np.eye(p), size=n).T
data_matrix = np.sqrt(signal_strength) * (true_vector[:, np.newaxis] @ latent_factors[np.newaxis, :]) + noise

# Run sparse PCA
distance, elapsed_time = spca(true_vector, data_matrix, p, k, n, sparsity_level, max_iter)

print(f"Distance: {distance}")
print(f"Elapsed time: {elapsed_time:.4f} seconds")
