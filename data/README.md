# Data Directory

This directory is used to store the MovieLens datasets.

## Dataset Download

The MovieLens datasets can be downloaded from the [GroupLens website](https://grouplens.org/datasets/movielens/):

### Available Datasets:

1. **MovieLens 100K** (Recommended for quick testing)
   - Download: https://files.grouplens.org/datasets/movielens/ml-100k.zip
   - Size: ~5 MB
   - 100,000 ratings from 943 users on 1,682 movies

2. **MovieLens 1M** (Recommended for experiments)
   - Download: https://files.grouplens.org/datasets/movielens/ml-1m.zip
   - Size: ~24 MB
   - 1 million ratings from 6,000 users on 4,000 movies

3. **MovieLens 10M**
   - Download: https://files.grouplens.org/datasets/movielens/ml-10m.zip
   - Size: ~63 MB
   - 10 million ratings from 72,000 users on 10,000 movies

4. **MovieLens 20M**
   - Download: https://files.grouplens.org/datasets/movielens/ml-20m.zip
   - Size: ~190 MB
   - 20 million ratings from 138,000 users on 27,000 movies

## Directory Structure

After downloading and extracting, your directory should look like:

```
data/
├── README.md
├── ml-100k/
│   ├── u.data
│   ├── u.item
│   ├── u.user
│   └── ...
├── ml-1m/
│   ├── ratings.dat
│   ├── movies.dat
│   ├── users.dat
│   └── ...
├── ml-10m/
│   └── ...
└── ml-20m/
    └── ...
```

## Note

The data files are not included in the repository due to their size. Please download them manually or use the automatic download feature in the notebooks.
