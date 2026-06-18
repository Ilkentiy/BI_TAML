import os
import torch

DATA_DIR = 'data/processed/enriched'
MODELS_DIR = 'models'
RESULTS_DIR = 'results/metrics'
PLOTS_DIR = 'results/plots'

TRAIN_PATH = os.path.join(DATA_DIR, 'train_enriched.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test_enriched.csv')
EMBEDDINGS_PATH = 'data/raw/articles_embeddings.pickle'
CLUSTER_PATH = os.path.join(DATA_DIR, 'user_clusters.pkl')

BATCH_SIZE = 2048
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 5
TEST_SIZE = 0.1
RANDOM_STATE = 42

TWO_TOWER_CONFIG = {
    'user_tower': {
        'emb_dim': 16,
        'hidden_dims': [256, 128, 64],
        'dropout': 0.3
    },
    'item_tower': {
        'content_dim': 250,
        'category_emb_dim': 32,
        'time_shift_emb_dim': 16,
        'hidden_dims': [256, 128, 64],
        'dropout': 0.3
    }
}

FEASE_CONFIG = {
    'context_dim': 6,
    'time_dim': 8,
    'content_dim': 250,
    'category_dim': 461,
    'time_shift_dim': 6,
    'hidden_dim': 256,
    'reduction': 16,
    'dropout': 0.3
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)