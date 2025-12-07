# Recommendation Systems Comparison - MovieLens Dataset

This repository compares different recommendation system algorithms using the MovieLens dataset. It serves as a comprehensive resource for understanding and implementing various collaborative filtering and deep learning-based recommendation approaches.

## Overview

This project implements and compares multiple recommendation system algorithms on the MovieLens dataset, including:

- **Collaborative Filtering**
  - User-based Collaborative Filtering
  - Item-based Collaborative Filtering
  - Matrix Factorization (SVD, SVD++)
  
- **Deep Learning Approaches**
  - Neural Collaborative Filtering (NCF)
  - Deep Matrix Factorization
  - Autoencoders for Collaborative Filtering
  
- **Hybrid Methods**
  - Content-based + Collaborative Filtering
  - Neural Graph Collaborative Filtering

## MovieLens Dataset

The [MovieLens dataset](https://grouplens.org/datasets/movielens/) is a widely-used benchmark for evaluating recommendation systems. It contains movie ratings from users and is available in several sizes:

- **MovieLens 100K**: 100,000 ratings from 943 users on 1,682 movies
- **MovieLens 1M**: 1 million ratings from 6,000 users on 4,000 movies
- **MovieLens 10M**: 10 million ratings from 72,000 users on 10,000 movies
- **MovieLens 20M**: 20 million ratings from 138,000 users on 27,000 movies

## State-of-the-Art Reproduction

For the highest accuracy published on the MovieLens dataset, we recommend exploring the reproduction work of **Neural Graph Collaborative Filtering (NGCF)**:

🔗 **[NGCF Reproduction Repository](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)**

The NGCF paper by Wang et al. (2019) has achieved state-of-the-art results on MovieLens and other datasets by leveraging graph neural networks to model user-item interactions. The reproduction repository provides:
- Complete implementation in TensorFlow
- Pre-trained models
- Detailed performance metrics on MovieLens-1M and MovieLens-10M
- Comprehensive documentation for reproduction

**Paper Reference:**
```
Wang, X., He, X., Wang, M., Feng, F., & Chua, T. S. (2019).
Neural graph collaborative filtering.
In Proceedings of the 42nd international ACM SIGIR conference on Research and development in Information Retrieval (pp. 165-174).
```

## Getting Started

### Prerequisites

```bash
python >= 3.7
pip install -r requirements.txt
```

### Installation

1. Clone this repository:
```bash
git clone https://github.com/Icy-Mint/Recommendation-Systems-Movie-Lens-Dataset.git
cd Recommendation-Systems-Movie-Lens-Dataset
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the MovieLens dataset:
```bash
# The dataset will be automatically downloaded when running the notebooks
# Or manually download from: https://grouplens.org/datasets/movielens/
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_collaborative_filtering.ipynb
│   ├── 03_matrix_factorization.ipynb
│   ├── 04_neural_collaborative_filtering.ipynb
│   └── 05_model_comparison.ipynb
├── src/
│   ├── data/
│   │   └── data_loader.py
│   ├── models/
│   │   ├── collaborative_filtering.py
│   │   ├── matrix_factorization.py
│   │   └── neural_models.py
│   └── utils/
│       ├── evaluation.py
│       └── visualization.py
└── data/
    └── README.md
```

## Usage

### Running Individual Models

```python
# Example: Collaborative Filtering
from src.models.collaborative_filtering import UserBasedCF

model = UserBasedCF()
model.fit(train_data)
predictions = model.predict(user_id, top_k=10)
```

### Comparing Models

Run the comparison notebook to evaluate all models:
```bash
jupyter notebook notebooks/05_model_comparison.ipynb
```

## Evaluation Metrics

The models are evaluated using standard recommendation metrics:

- **Precision@K**: Proportion of recommended items that are relevant
- **Recall@K**: Proportion of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Hit Rate@K**: Whether any recommended item is relevant
- **MRR**: Mean Reciprocal Rank

## Results

Performance comparison on MovieLens-1M dataset:

| Model | Precision@10 | Recall@10 | NDCG@10 | Hit Rate@10 |
|-------|--------------|-----------|---------|-------------|
| User-based CF | - | - | - | - |
| Item-based CF | - | - | - | - |
| SVD | - | - | - | - |
| NCF | - | - | - | - |
| **NGCF (SOTA)** | **0.157** | **0.328** | **0.342** | **0.726** |

*Note: NGCF results are from the original paper. Other results will be added as implementations are completed.*

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

1. Wang, X., He, X., Wang, M., Feng, F., & Chua, T. S. (2019). Neural graph collaborative filtering. SIGIR 2019.
2. He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural collaborative filtering. WWW 2017.
3. Koren, Y. (2008). Factorization meets the neighborhood: a multifaceted collaborative filtering model. KDD 2008.
4. Harper, F. M., & Konstan, J. A. (2015). The movielens datasets: History and context. ACM TIIS.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- GroupLens Research for providing the MovieLens dataset
- The authors of NGCF for their state-of-the-art work and open-source implementation
- The recommendation systems research community