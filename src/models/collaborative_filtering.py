"""
Collaborative Filtering models for recommendation systems.

This module implements user-based and item-based collaborative filtering algorithms.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class UserBasedCF:
    """
    User-based Collaborative Filtering.
    
    Recommends items based on the preferences of similar users.
    """
    
    def __init__(self, k_neighbors: int = 20, similarity_metric: str = 'cosine'):
        """
        Initialize User-based CF model.
        
        Args:
            k_neighbors: Number of similar users to consider
            similarity_metric: Similarity metric to use ('cosine')
        """
        self.k_neighbors = k_neighbors
        self.similarity_metric = similarity_metric
        self.user_item_matrix = None
        self.user_similarity = None
        self.user_ids = None
        self.item_ids = None
        self.user_id_to_idx = None
    
    def fit(self, ratings: pd.DataFrame):
        """
        Fit the model on training data.
        
        Args:
            ratings: DataFrame with columns ['user_id', 'item_id', 'rating']
        """
        # Create user-item matrix
        self.user_ids = sorted(ratings['user_id'].unique())
        self.item_ids = sorted(ratings['item_id'].unique())
        
        user_id_map = {uid: idx for idx, uid in enumerate(self.user_ids)}
        item_id_map = {iid: idx for idx, iid in enumerate(self.item_ids)}
        
        # Store mapping for efficient lookup
        self.user_id_to_idx = user_id_map
        
        rows = ratings['user_id'].map(user_id_map)
        cols = ratings['item_id'].map(item_id_map)
        data = ratings['rating'].values
        
        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids))
        ).toarray()
        
        # Compute user similarity matrix
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        # Set self-similarity to 0 to avoid recommending based on user's own ratings
        np.fill_diagonal(self.user_similarity, 0)
    
    def predict(self, user_id: int, top_k: int = 10) -> List[int]:
        """
        Generate top-K recommendations for a user.
        
        Args:
            user_id: User ID to generate recommendations for
            top_k: Number of recommendations to generate
            
        Returns:
            List of recommended item IDs
        """
        if user_id not in self.user_id_to_idx:
            return []
        
        user_idx = self.user_id_to_idx[user_id]
        
        # Get k most similar users using argpartition for better performance
        if self.k_neighbors < len(self.user_similarity[user_idx]):
            similar_users_idx = np.argpartition(
                self.user_similarity[user_idx], 
                -self.k_neighbors
            )[-self.k_neighbors:]
            # Sort the top-k for consistent ordering
            similar_users = similar_users_idx[np.argsort(self.user_similarity[user_idx][similar_users_idx])[::-1]]
        else:
            similar_users = np.argsort(self.user_similarity[user_idx])[::-1][:self.k_neighbors]
        
        # Get items the user hasn't rated
        user_ratings = self.user_item_matrix[user_idx]
        unrated_items = np.where(user_ratings == 0)[0]
        
        # Predict ratings for unrated items
        predictions = {}
        for item_idx in unrated_items:
            # Weighted average of similar users' ratings
            numerator = 0
            denominator = 0
            for similar_user_idx in similar_users:
                if self.user_item_matrix[similar_user_idx, item_idx] > 0:
                    similarity = self.user_similarity[user_idx, similar_user_idx]
                    rating = self.user_item_matrix[similar_user_idx, item_idx]
                    numerator += similarity * rating
                    denominator += abs(similarity)
            
            if denominator > 0:
                predictions[self.item_ids[item_idx]] = numerator / denominator
        
        # Sort by predicted rating and return top-K
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_predictions[:top_k]]


class ItemBasedCF:
    """
    Item-based Collaborative Filtering.
    
    Recommends items similar to those the user has liked.
    """
    
    def __init__(self, k_neighbors: int = 20, similarity_metric: str = 'cosine'):
        """
        Initialize Item-based CF model.
        
        Args:
            k_neighbors: Number of similar items to consider
            similarity_metric: Similarity metric to use ('cosine')
        """
        self.k_neighbors = k_neighbors
        self.similarity_metric = similarity_metric
        self.user_item_matrix = None
        self.item_similarity = None
        self.user_ids = None
        self.item_ids = None
        self.user_id_to_idx = None
    
    def fit(self, ratings: pd.DataFrame):
        """
        Fit the model on training data.
        
        Args:
            ratings: DataFrame with columns ['user_id', 'item_id', 'rating']
        """
        # Create user-item matrix
        self.user_ids = sorted(ratings['user_id'].unique())
        self.item_ids = sorted(ratings['item_id'].unique())
        
        user_id_map = {uid: idx for idx, uid in enumerate(self.user_ids)}
        item_id_map = {iid: idx for idx, iid in enumerate(self.item_ids)}
        
        # Store mapping for efficient lookup
        self.user_id_to_idx = user_id_map
        
        rows = ratings['user_id'].map(user_id_map)
        cols = ratings['item_id'].map(item_id_map)
        data = ratings['rating'].values
        
        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids))
        ).toarray()
        
        # Compute item similarity matrix
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        # Set self-similarity to 0
        np.fill_diagonal(self.item_similarity, 0)
    
    def predict(self, user_id: int, top_k: int = 10) -> List[int]:
        """
        Generate top-K recommendations for a user.
        
        Args:
            user_id: User ID to generate recommendations for
            top_k: Number of recommendations to generate
            
        Returns:
            List of recommended item IDs
        """
        if user_id not in self.user_id_to_idx:
            return []
        
        user_idx = self.user_id_to_idx[user_id]
        user_ratings = self.user_item_matrix[user_idx]
        
        # Get items the user has rated
        rated_items = np.where(user_ratings > 0)[0]
        unrated_items = np.where(user_ratings == 0)[0]
        
        # Predict ratings for unrated items
        predictions = {}
        for item_idx in unrated_items:
            # Find k most similar items that user has rated
            item_similarities = self.item_similarity[item_idx]
            
            numerator = 0
            denominator = 0
            for rated_item_idx in rated_items:
                similarity = item_similarities[rated_item_idx]
                if similarity > 0:
                    rating = user_ratings[rated_item_idx]
                    numerator += similarity * rating
                    denominator += abs(similarity)
            
            if denominator > 0:
                predictions[self.item_ids[item_idx]] = numerator / denominator
        
        # Sort by predicted rating and return top-K
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_predictions[:top_k]]
