# Bi-TAML: Two-Tower Neural Architecture for Cold-Start News Recommendation

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> Two-Tower neural architecture for cold-start news recommendation using temporal features and user clustering.

## Background

News recommendation systems face significant challenges when serving new users with no historical interaction data. Traditional collaborative filtering and sequential recommendation methods fail in cold-start scenarios where user history is unavailable.

Bi-TAML (Bilateral Time-Aware Meta-Learning) addresses this problem through:
- Two-Tower Architecture: Separate encoders for user context and article content
- Temporal Features: Time-specific bins (3-hour windows) and time-shift bins (article age)
- User Clustering: K-means clustering based on contextual features
- Cold-Start Focus: Works without user interaction history

---

## Installation

```bash
git clone https://github.com/ilkentiy/Bi_TAML
cd bitaml
pip install -r requirements.txt
```
## Project Structure

```
Bi_TAML/
├── data/
│   ├── raw/
│   │   ├── clicks/
│   │   │   └── clicks_hour_*.csv
│   │   ├── articles_metadata.csv
│   │   └── articles_embeddings.pickle
│   └── processed/
│       └── enriched/
│           ├── train_enriched.csv
│           ├── test_enriched.csv
│           └── user_clusters.pkl
├── src/
│   ├── data/
│   │   ├── enrich_data.py
│   │   └── create_clusters.py
│   └── models/
│       ├── two_tower_ranker.py
│       └── fease_model.py
├── scripts/
│   ├── config.py
│   ├── train_two_tower.py
│   ├── train_fease.py
│   ├── train_popularity.py
│   └── train_random.py
├── notebooks/
├── models/
│   ├── two_tower_ranker.pth
│   └── fease_ranker.pth
├── results/
│   ├── metrics/
│   │   └── metrics.csv
│   └── plots/
│       ├── training_history.png
│       └── model_comparison.png
├── requirements.txt
└── README.md
```

## License
MIT License