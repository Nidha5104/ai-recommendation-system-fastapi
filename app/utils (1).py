"""
utils.py
--------
Shared utility / helper functions used across the recommendation system.
Keeping these separate from recommender.py makes unit-testing trivial and
avoids circular imports.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


def compute_cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise cosine similarity for every row in *matrix*.

    Args:
        matrix: 2-D NumPy array where each row is a vector (e.g. a user or
                item profile).

    Returns:
        Square similarity matrix of shape (n, n), values in [-1, 1].
    """
    return cosine_similarity(matrix)


def get_top_n_indices(similarity_row: np.ndarray, n: int, exclude_index: int) -> List[int]:
    """
    Return the indices of the top-n most similar entries in *similarity_row*,
    excluding the entity itself (self-similarity = 1.0 is always the highest).

    Args:
        similarity_row: 1-D array of similarity scores.
        n:              Number of top results to return.
        exclude_index:  Index of the query entity (excluded from results).

    Returns:
        List of *n* integer indices, sorted descending by similarity.
    """
    # argsort ascending → reverse → skip the first occurrence of exclude_index
    sorted_indices = np.argsort(similarity_row)[::-1]
    filtered = [idx for idx in sorted_indices if idx != exclude_index]
    return filtered[:n]


def scores_to_dict(
    indices: List[int],
    id_list: List[int],
    similarity_row: np.ndarray,
) -> List[dict]:
    """
    Convert a list of matrix indices into human-readable score dictionaries.

    Args:
        indices:        Ordered list of matrix indices (top-n results).
        id_list:        Mapping from matrix index → real entity ID.
        similarity_row: Full similarity score array (used to fetch scores).

    Returns:
        List of dicts: [{"id": <entity_id>, "score": <float>}, ...]
    """
    return [
        {"id": int(id_list[i]), "score": round(float(similarity_row[i]), 4)}
        for i in indices
    ]


def normalize_matrix(matrix: pd.DataFrame) -> np.ndarray:
    """
    Mean-center each user's ratings (subtract the user's mean rating).
    This improves cosine similarity by removing user-level rating bias
    (e.g. a user who always rates 5 vs. a user who rates 3 for equivalent items).

    Zero rows (users with no interactions) are left as-is.

    Args:
        matrix: User-item DataFrame (rows=users, cols=items).

    Returns:
        NumPy array of the same shape, mean-centered per row.
    """
    arr = matrix.values.astype(float)
    row_means = np.true_divide(
        arr.sum(axis=1),
        (arr != 0).sum(axis=1),
        where=(arr != 0).sum(axis=1) != 0,
        out=np.zeros(arr.shape[0]),
    )
    # Subtract mean only from non-zero entries
    mask = arr != 0
    arr[mask] -= np.repeat(row_means, arr.shape[1]).reshape(arr.shape)[mask]
    return arr
