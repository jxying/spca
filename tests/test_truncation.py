import numpy as np
from src.truncation import truncation

def test_truncation_basic():
    vector = np.array([1, -3, 2, 4, -5])
    # Keep the 2 largest absolute values: 4 and -5, so others should be zero
    result = truncation(vector, 2)
    expected = np.array([0, 0, 0, 4, -5])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"

def test_truncation_full():
    vector = np.array([1, 2, 3])
    # Keep all elements
    result = truncation(vector, 3)
    expected = np.array([1, 2, 3])
    assert np.allclose(result, expected)

def test_truncation_none():
    vector = np.array([1, 2, 3])
    # Keep zero elements: all should be zero
    result = truncation(vector, 0)
    expected = np.array([0, 0, 0])
    assert np.allclose(result, expected)
