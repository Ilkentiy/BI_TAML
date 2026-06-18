import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import json
import argparse
from datetime import datetime

from scripts.config import *


def train(args):
    print("Training Popularity baseline")
    
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    print(f"Train: {len(train_df):,} rows, Test: {len(test_df):,} rows")
    
    article_popularity = train_df.groupby('click_article_id')['label'].mean()
    print(f"Articles with popularity: {len(article_popularity):,}")
    
    test_df['pred_score'] = test_df['click_article_id'].map(article_popularity)
    test_df['pred_score'] = test_df['pred_score'].fillna(0.5)
    
    auc = roc_auc_score(test_df['label'], test_df['pred_score'])
    print(f"Test AUC: {auc:.4f}")
    
    results_file = os.path.join(
        args.results_dir, 
        f'popularity_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    with open(results_file, 'w') as f:
        json.dump({
            'model': 'Popularity',
            'test_auc': auc
        }, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    return {'auc': auc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Popularity baseline')
    parser.add_argument('--train_path', type=str, default=TRAIN_PATH)
    parser.add_argument('--test_path', type=str, default=TEST_PATH)
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR)
    
    args = parser.parse_args()
    train(args)