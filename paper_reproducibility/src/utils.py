import numpy as np


def get_new_sequence_mask_np(ids):
    ids = np.array(ids)
    mask = np.empty(len(ids), dtype=bool)
    mask[0] = True
    mask[1:] = ids[1:] != ids[:-1]
    return mask


def get_last_sequence_mask_np(ids):
    ids = np.array(ids)
    mask = np.empty(len(ids), dtype=bool)
    mask[:-1] = ids[:-1] != ids[1:]
    mask[-1] = True
    return mask


def temperatured_softmax(x, T=1.0):
    """
    Compute the temperature-scaled softmax function.

    Parameters:
    - x: Input array (1D or 2D)
    - T: Temperature parameter (default 1.0 for standard softmax)

    Returns:
    - Softmax probabilities with temperature scaling
    """
    if T == 0:
        # return zeros except max element
        zeros_like = np.zeros_like(x)
        zeros_like[np.argmax(x)] = 1
        return zeros_like

    # Subtract max value in x for numerical stability
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x / T)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def multiply_non_diagonal(matrix, alpha):
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    matrix[mask] *= alpha
    return matrix
