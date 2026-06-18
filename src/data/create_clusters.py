import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import os
import argparse

def create_user_clusters(train_path, test_path, output_dir, n_clusters=50):
    """Cluster users by context features: environment, device, os, country, region, referrer"""
    os.makedirs(output_dir, exist_ok=True)
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    all_users = pd.concat([train, test], ignore_index=True)
    
    # Context features used for clustering
    context_features = ['click_environment', 'click_deviceGroup', 
                        'click_os', 'click_country', 'click_region',
                        'click_referrer_type']
    
    # Check which features exist
    existing_features = [f for f in context_features if f in all_users.columns]
    
    if not existing_features:
        print("ERROR: No context features found for clustering!")
        print(f"Available columns: {all_users.columns.tolist()}")
        return {}
    
    print(f"Using features: {existing_features}")
    
    # Aggregate by user (mean values)
    user_features = all_users.groupby('user_id')[existing_features].mean()
    
    # Check if we have enough users
    if len(user_features) < n_clusters:
        print(f"Warning: Only {len(user_features)} users, reducing clusters to {len(user_features)}")
        n_clusters = max(2, len(user_features))
    
    # Normalize
    scaler = StandardScaler()
    user_features_scaled = scaler.fit_transform(user_features)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    user_clusters = kmeans.fit_predict(user_features_scaled)
    
    # Map user_id -> cluster_id
    cluster_map = dict(zip(user_features.index, user_clusters))
    
    # Save cluster data
    with open(os.path.join(output_dir, 'user_clusters.pkl'), 'wb') as f:
        pickle.dump({
            'cluster_map': cluster_map,
            'kmeans': kmeans,
            'scaler': scaler
        }, f)
    
    # Print statistics
    cluster_sizes = pd.Series(user_clusters).value_counts().sort_index()
    print(f"Saved clusters: {output_dir}/user_clusters.pkl ({n_clusters} clusters)")
    print(f"Cluster sizes: {cluster_sizes.tolist()}")
    print(f"Min cluster size: {cluster_sizes.min()}, Max: {cluster_sizes.max()}")
    
    return cluster_map

def main():
    parser = argparse.ArgumentParser(description='Create user clusters')
    parser.add_argument('--train', type=str,
                       default='data/processed/enriched/train_enriched.csv',
                       help='Train enriched CSV path')
    parser.add_argument('--test', type=str,
                       default='data/processed/enriched/test_enriched.csv',
                       help='Test enriched CSV path')
    parser.add_argument('--output_dir', type=str,
                       default='data/processed/enriched',
                       help='Output directory')
    parser.add_argument('--n_clusters', type=int, default=50,
                       help='Number of clusters (default: 50)')
    
    args = parser.parse_args()
    create_user_clusters(args.train, args.test, args.output_dir, args.n_clusters)

if __name__ == '__main__':
    main()