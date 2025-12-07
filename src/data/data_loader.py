"""
Data loader module for MovieLens datasets.

This module provides utilities to load and preprocess MovieLens datasets
of different sizes (100K, 1M, 10M, 20M).
"""

import os
import pandas as pd
import numpy as np
from urllib.request import urlretrieve
from zipfile import ZipFile
from typing import Tuple, Optional


class MovieLensLoader:
    """
    Loader for MovieLens datasets.
    
    Supports automatic downloading and loading of MovieLens 100K, 1M, 10M, and 20M datasets.
    """
    
    DATASET_URLS = {
        '100k': 'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
        '1m': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
        '10m': 'https://files.grouplens.org/datasets/movielens/ml-10m.zip',
        '20m': 'https://files.grouplens.org/datasets/movielens/ml-20m.zip',
    }
    
    def __init__(self, data_dir: str = './data'):
        """
        Initialize the MovieLens data loader.
        
        Args:
            data_dir: Directory where datasets will be stored
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_dataset(self, dataset_size: str = '100k') -> str:
        """
        Download a MovieLens dataset if not already present.
        
        Args:
            dataset_size: Size of dataset ('100k', '1m', '10m', '20m')
            
        Returns:
            Path to the extracted dataset directory
        """
        if dataset_size not in self.DATASET_URLS:
            raise ValueError(f"Invalid dataset size. Choose from {list(self.DATASET_URLS.keys())}")
        
        dataset_path = os.path.join(self.data_dir, f'ml-{dataset_size}')
        
        if os.path.exists(dataset_path):
            print(f"Dataset ml-{dataset_size} already exists at {dataset_path}")
            return dataset_path
        
        print(f"Downloading MovieLens {dataset_size} dataset...")
        url = self.DATASET_URLS[dataset_size]
        zip_path = os.path.join(self.data_dir, f'ml-{dataset_size}.zip')
        
        urlretrieve(url, zip_path)
        
        print(f"Extracting dataset...")
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        
        os.remove(zip_path)
        print(f"Dataset downloaded and extracted to {dataset_path}")
        
        return dataset_path
    
    def load_ratings(self, dataset_size: str = '100k') -> pd.DataFrame:
        """
        Load ratings data from a MovieLens dataset.
        
        Args:
            dataset_size: Size of dataset ('100k', '1m', '10m', '20m')
            
        Returns:
            DataFrame with columns: user_id, item_id, rating, timestamp
        """
        dataset_path = self.download_dataset(dataset_size)
        
        if dataset_size == '100k':
            # MovieLens 100K uses tab-separated format
            ratings_file = os.path.join(dataset_path, 'u.data')
            df = pd.read_csv(
                ratings_file,
                sep='\t',
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                engine='python'
            )
        elif dataset_size == '1m':
            # MovieLens 1M uses :: separator
            ratings_file = os.path.join(dataset_path, 'ratings.dat')
            df = pd.read_csv(
                ratings_file,
                sep='::',
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                engine='python'
            )
        else:
            # MovieLens 10M and 20M use :: or , separator
            ratings_file = os.path.join(dataset_path, 'ratings.csv')
            if os.path.exists(ratings_file):
                df = pd.read_csv(ratings_file)
                df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
            else:
                ratings_file = os.path.join(dataset_path, 'ratings.dat')
                df = pd.read_csv(
                    ratings_file,
                    sep='::',
                    names=['user_id', 'item_id', 'rating', 'timestamp'],
                    engine='python'
                )
        
        return df
    
    def load_movies(self, dataset_size: str = '100k') -> pd.DataFrame:
        """
        Load movie metadata from a MovieLens dataset.
        
        Args:
            dataset_size: Size of dataset ('100k', '1m', '10m', '20m')
            
        Returns:
            DataFrame with movie information
        """
        dataset_path = os.path.join(self.data_dir, f'ml-{dataset_size}')
        
        if dataset_size == '100k':
            movies_file = os.path.join(dataset_path, 'u.item')
            df = pd.read_csv(
                movies_file,
                sep='|',
                encoding='latin-1',
                names=['item_id', 'title', 'release_date', 'video_release_date',
                       'imdb_url'] + [f'genre_{i}' for i in range(19)],
                engine='python'
            )
        elif dataset_size == '1m':
            movies_file = os.path.join(dataset_path, 'movies.dat')
            df = pd.read_csv(
                movies_file,
                sep='::',
                names=['item_id', 'title', 'genres'],
                encoding='latin-1',
                engine='python'
            )
        else:
            movies_file = os.path.join(dataset_path, 'movies.csv')
            df = pd.read_csv(movies_file)
            if 'movieId' in df.columns:
                df.rename(columns={'movieId': 'item_id'}, inplace=True)
        
        return df
    
    def train_test_split(
        self,
        ratings: pd.DataFrame,
        test_size: float = 0.2,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split ratings into train and test sets.
        
        Args:
            ratings: Ratings DataFrame
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Shuffle the data
        ratings_shuffled = ratings.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Split
        split_idx = int(len(ratings_shuffled) * (1 - test_size))
        train_df = ratings_shuffled[:split_idx]
        test_df = ratings_shuffled[split_idx:]
        
        return train_df, test_df


def get_data_statistics(ratings: pd.DataFrame) -> dict:
    """
    Get statistics about the ratings dataset.
    
    Args:
        ratings: Ratings DataFrame
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'num_ratings': len(ratings),
        'num_users': ratings['user_id'].nunique(),
        'num_items': ratings['item_id'].nunique(),
        'sparsity': 1 - (len(ratings) / (ratings['user_id'].nunique() * ratings['item_id'].nunique())),
        'min_rating': ratings['rating'].min(),
        'max_rating': ratings['rating'].max(),
        'mean_rating': ratings['rating'].mean(),
        'median_rating': ratings['rating'].median(),
    }
    
    return stats
