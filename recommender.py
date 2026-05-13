"""
recommender.py
--------------
Core recommendation engine.

Two complementary strategies are implemented:

1. User-Based Collaborative Filtering  →  get_recommendations(user_id)
   ─ Find users most similar to the target user (cosine similarity on the
     mean-centered user-item matrix).
   ─ Aggregate their ratings (weighted by similarity) to predict scores for
     items the target user has NOT yet interacted with.
   ─ Return the top-N predicted items.

2. Item-Based Cosine Similarity  →  get_similar_items(item_id)
   ─ Transpose the matrix so rows = items.
   ─ Compute item-item cosine similarity.
   ─ Return the top-N most similar items to the query item.

The RecommendationEngine class is designed to be instantiated once at
application startup (expensive matrix computations run only once) and then
reused for every API request.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any

from app.data_loader import get_preprocessed_data
from app.utils import (
    compute_cosine_similarity,
    get_top_n_indices,
    scores_to_dict,
    normalize_matrix,
)

# Number of recommendations / similar items to return
TOP_N = 5


class RecommendationEngine:
    """
    Encapsulates all recommendation logic.

    Attributes:
        matrix        (pd.DataFrame): user-item interaction matrix
        user_ids      (list[int]):    ordered user IDs (row index)
        item_ids      (list[int]):    ordered item IDs (column index)
        user_sim      (np.ndarray):   user×user cosine similarity matrix
        item_sim      (np.ndarray):   item×item cosine similarity matrix
        user_index    (dict):         user_id → row index lookup
        item_index    (dict):         item_id → column index lookup
    """

    def __init__(self, data_path: str = None):
        """
        Load data and pre-compute similarity matrices.

        Args:
            data_path: Optional override for the CSV file path.
        """
        # ── 1. Load and build the user-item matrix ─────────────────────────
        kwargs = {"filepath": data_path} if data_path else {}
        self.matrix, self.user_ids, self.item_ids = get_preprocessed_data(**kwargs)

        # ── 2. Fast lookup dictionaries ────────────────────────────────────
        self.user_index: Dict[int, int] = {uid: i for i, uid in enumerate(self.user_ids)}
        self.item_index: Dict[int, int] = {iid: i for i, iid in enumerate(self.item_ids)}

        # ── 3. Pre-compute similarity matrices ─────────────────────────────
        #   User similarity: mean-center rows first to reduce rating-scale bias
        normalized = normalize_matrix(self.matrix)
        self.user_sim: np.ndarray = compute_cosine_similarity(normalized)

        #   Item similarity: transpose → items as rows, users as columns
        self.item_sim: np.ndarray = compute_cosine_similarity(
            self.matrix.values.T.astype(float)
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def get_recommendations(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Return the top-N item recommendations for a given user via
        user-based collaborative filtering.

        Steps:
          a) Find the user's row in the similarity matrix.
          b) Weight neighbour ratings by similarity score.
          c) Predict scores only for items the user has NOT rated.
          d) Rank and return top-N.

        Args:
            user_id: Target user ID.

        Returns:
            List of dicts: [{"item_id": int, "predicted_score": float}, ...]

        Raises:
            ValueError: If user_id is not found in the dataset.
        """
        if user_id not in self.user_index:
            raise ValueError(
                f"user_id={user_id} not found. "
                f"Valid user IDs: {sorted(self.user_ids)}"
            )

        u_idx = self.user_index[user_id]
        sim_scores = self.user_sim[u_idx]           # similarity to all other users
        user_ratings = self.matrix.values[u_idx]    # this user's rating vector

        # Items the user has already interacted with (rated > 0)
        rated_mask = user_ratings > 0

        # Weighted sum prediction:
        #   predicted_score[item] = Σ(sim[u, v] * rating[v, item]) / Σ|sim[u, v]|
        #   Only sum over users who actually rated the item.
        matrix_vals = self.matrix.values           # shape (n_users, n_items)
        sim_col = sim_scores.reshape(-1, 1)        # shape (n_users, 1)

        # Numerator: weighted ratings
        weighted_ratings = (sim_col * matrix_vals).sum(axis=0)

        # Denominator: sum of absolute similarities for users who rated each item
        rated_by = (matrix_vals > 0).astype(float)
        sim_sums = (np.abs(sim_col) * rated_by).sum(axis=0)

        # Avoid division by zero
        predicted = np.where(sim_sums > 0, weighted_ratings / sim_sums, 0.0)

        # Zero out already-rated items so we only recommend NEW items
        predicted[rated_mask] = 0.0

        # Get top-N item indices
        top_indices = np.argsort(predicted)[::-1][:TOP_N]

        return [
            {
                "item_id": int(self.item_ids[i]),
                "predicted_score": round(float(predicted[i]), 4),
            }
            for i in top_indices
            if predicted[i] > 0  # omit items with zero predicted interest
        ]

    def get_similar_items(self, item_id: int) -> List[Dict[str, Any]]:
        """
        Return the top-N items most similar to the given item via
        item-item cosine similarity.

        Args:
            item_id: Query item ID.

        Returns:
            List of dicts: [{"item_id": int, "similarity_score": float}, ...]

        Raises:
            ValueError: If item_id is not found in the dataset.
        """
        if item_id not in self.item_index:
            raise ValueError(
                f"item_id={item_id} not found. "
                f"Valid item IDs: {sorted(self.item_ids)}"
            )

        i_idx = self.item_index[item_id]
        sim_row = self.item_sim[i_idx]             # similarity to all other items

        top_indices = get_top_n_indices(sim_row, n=TOP_N, exclude_index=i_idx)
        results = scores_to_dict(top_indices, self.item_ids, sim_row)

        return [{"item_id": r["id"], "similarity_score": r["score"]} for r in results]
