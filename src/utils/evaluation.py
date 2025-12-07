"""
Evaluation metrics for recommendation systems.

This module provides common evaluation metrics used to assess
the performance of recommendation systems.
"""

import numpy as np
from typing import List, Dict, Set
import pandas as pd


def precision_at_k(recommended: List, relevant: Set, k: int = 10) -> float:
    """
    Calculate Precision@K.
    
    Precision@K measures the proportion of recommended items in the top-K set 
    that are relevant.
    
    Args:
        recommended: List of recommended item IDs (ordered by relevance)
        relevant: Set of relevant item IDs
        k: Number of top recommendations to consider
        
    Returns:
        Precision@K score
    """
    recommended_at_k = recommended[:k]
    num_relevant = len([item for item in recommended_at_k if item in relevant])
    return num_relevant / k if k > 0 else 0.0


def recall_at_k(recommended: List, relevant: Set, k: int = 10) -> float:
    """
    Calculate Recall@K.
    
    Recall@K measures the proportion of relevant items that are successfully 
    recommended in the top-K set.
    
    Args:
        recommended: List of recommended item IDs (ordered by relevance)
        relevant: Set of relevant item IDs
        k: Number of top recommendations to consider
        
    Returns:
        Recall@K score
    """
    if len(relevant) == 0:
        return 0.0
    
    recommended_at_k = recommended[:k]
    num_relevant = len([item for item in recommended_at_k if item in relevant])
    return num_relevant / len(relevant)


def ndcg_at_k(recommended: List, relevant: Set, k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K).
    
    NDCG@K measures the ranking quality of recommended items, giving higher
    scores to relevant items appearing earlier in the recommendation list.
    
    Args:
        recommended: List of recommended item IDs (ordered by relevance)
        relevant: Set of relevant item IDs
        k: Number of top recommendations to consider
        
    Returns:
        NDCG@K score
    """
    recommended_at_k = recommended[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(recommended_at_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because index starts at 0
    
    # Calculate IDCG (Ideal DCG)
    idcg = 0.0
    for i in range(min(len(relevant), k)):
        idcg += 1.0 / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: List, relevant: Set, k: int = 10) -> float:
    """
    Calculate Hit Rate@K.
    
    Hit Rate@K is a binary metric that indicates whether at least one relevant
    item appears in the top-K recommendations.
    
    Args:
        recommended: List of recommended item IDs (ordered by relevance)
        relevant: Set of relevant item IDs
        k: Number of top recommendations to consider
        
    Returns:
        Hit Rate@K score (1.0 if hit, 0.0 otherwise)
    """
    recommended_at_k = recommended[:k]
    return 1.0 if any(item in relevant for item in recommended_at_k) else 0.0


def mean_reciprocal_rank(recommended: List, relevant: Set) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    MRR measures how far down the first relevant item appears in the 
    recommendation list.
    
    Args:
        recommended: List of recommended item IDs (ordered by relevance)
        relevant: Set of relevant item IDs
        
    Returns:
        MRR score
    """
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_model(
    predictions: Dict[int, List[int]],
    ground_truth: Dict[int, Set[int]],
    k_values: List[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    Evaluate a recommendation model using multiple metrics.
    
    Args:
        predictions: Dictionary mapping user IDs to lists of recommended item IDs
        ground_truth: Dictionary mapping user IDs to sets of relevant item IDs
        k_values: List of K values to evaluate at
        
    Returns:
        DataFrame with evaluation results for each K value
    """
    results = []
    
    for k in k_values:
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        hit_rate_scores = []
        mrr_scores = []
        
        for user_id in predictions:
            if user_id not in ground_truth:
                continue
            
            recommended = predictions[user_id]
            relevant = ground_truth[user_id]
            
            if len(relevant) == 0:
                continue
            
            precision_scores.append(precision_at_k(recommended, relevant, k))
            recall_scores.append(recall_at_k(recommended, relevant, k))
            ndcg_scores.append(ndcg_at_k(recommended, relevant, k))
            hit_rate_scores.append(hit_rate_at_k(recommended, relevant, k))
            mrr_scores.append(mean_reciprocal_rank(recommended, relevant))
        
        results.append({
            'K': k,
            'Precision@K': np.mean(precision_scores) if precision_scores else 0.0,
            'Recall@K': np.mean(recall_scores) if recall_scores else 0.0,
            'NDCG@K': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            'Hit Rate@K': np.mean(hit_rate_scores) if hit_rate_scores else 0.0,
            'MRR': np.mean(mrr_scores) if mrr_scores else 0.0,
        })
    
    return pd.DataFrame(results)


def compute_coverage(predictions: Dict[int, List[int]], num_items: int) -> float:
    """
    Calculate catalog coverage - the percentage of items that are recommended.
    
    Args:
        predictions: Dictionary mapping user IDs to lists of recommended item IDs
        num_items: Total number of items in the catalog
        
    Returns:
        Coverage score (0 to 1)
    """
    recommended_items = set()
    for user_recommendations in predictions.values():
        recommended_items.update(user_recommendations)
    
    return len(recommended_items) / num_items if num_items > 0 else 0.0
