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
    print("Training Random baseline")
    
    test_df = pd.read_csv(args.test_path)
    print(f"Test: {len(test_df):,} rows")
    
    np.random.seed(args.random_state)
    test_df['pred_score'] = np.random.random(len(test_df))
    
    auc = roc_auc_score(test_df['label'], test_df['pred_score'])
    print(f"Test AUC: {auc:.4f}")
    
    results_file = os.path.join(
        args.results_dir, 
        f'random_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    with open(results_file, 'w') as f:
        json.dump({
            'model': 'Random',
            'test_auc': auc
        }, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    return {'auc': auc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Random baseline')
    parser.add_argument('--test_path', type=str, default=TEST_PATH)
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR)
    parser.add_argument('--random_state', type=int, default=RANDOM_STATE)
    
    args = parser.parse_args()
    train(args)