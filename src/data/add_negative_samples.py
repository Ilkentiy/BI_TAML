import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
from bisect import bisect_left

def create_balanced_windows(input_path,
                           metadata_path='articles_metadata.csv',
                           output_path='data/processed/split_positive_negative/train.csv',
                           window_size_hours=3,
                           max_news_age_days=4,
                           window_length=100):
    """
    Create balanced windows with positive and negative samples for each user-time window.
    Each window contains up to 100 samples (positive + negative).
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load data
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} records from {input_path}")
    
    # Load article metadata for negative sampling
    metadata = pd.read_csv(metadata_path)
    metadata = metadata.sort_values('created_at_ts').reset_index(drop=True)
    news_created_times = metadata['created_at_ts'].values
    news_ids_sorted = metadata['article_id'].values
    
    # Keep only necessary columns
    cols = ['user_id', 'session_id', 'click_article_id', 'click_timestamp']
    df = df[cols].copy()
    df['click_timestamp'] = pd.to_datetime(df['click_timestamp'])
    df = df.sort_values(['user_id', 'click_timestamp']).reset_index(drop=True)
    
    # Create time windows
    df['time_window'] = df.groupby('user_id')['click_timestamp'].transform(
        lambda x: (x - x.min()).dt.total_seconds() // (window_size_hours * 3600)
    )
    
    grouped = df.groupby(['user_id', 'time_window'])
    total_windows = grouped.ngroups
    print(f"Total groups: {total_windows:,}")
    
    max_news_age_ms = max_news_age_days * 24 * 3600 * 1000
    existing_pairs = set(zip(df['user_id'], df['click_article_id']))
    
    all_records = []
    
    # Process each user-time window
    for (user_id, window), group in tqdm(grouped, desc="Processing groups", total=total_windows):
        # Collect positive records
        positive_records = []
        for idx, row in group.iterrows():
            positive_records.append({
                'user_id': int(user_id),
                'session_id': int(row['session_id']),
                'click_article_id': int(row['click_article_id']),
                'click_timestamp': int(row['click_timestamp'].timestamp() * 1000),
                'label': 1
            })
        
        n_positive = len(positive_records)
        
        if n_positive == 0:
            continue
        
        # Calculate how many negatives we need
        n_negative_needed = window_length - n_positive
        
        # If more than window_length positives, keep only first window_length
        if n_positive > window_length:
            positive_records = positive_records[:window_length]
            n_positive = window_length
            n_negative_needed = 0
        
        # Generate negative samples
        negative_records = []
        user_clicks = set(df[df['user_id'] == user_id]['click_article_id'])
        
        if n_negative_needed > 0:
            for pos_record in positive_records:
                click_time = pos_record['click_timestamp']
                
                min_created_time = click_time - max_news_age_ms
                start_idx = bisect_left(news_created_times, min_created_time)
                end_idx = bisect_left(news_created_times, click_time + 1)
                
                n_suitable = end_idx - start_idx
                if n_suitable == 0:
                    continue
                
                found = False
                attempts = 0
                max_attempts = min(10, n_suitable)
                
                while not found and attempts < max_attempts and len(negative_records) < n_negative_needed:
                    attempts += 1
                    random_idx = start_idx + np.random.randint(0, n_suitable)
                    negative_news_id = news_ids_sorted[random_idx]
                    
                    if negative_news_id not in user_clicks and (user_id, negative_news_id) not in existing_pairs:
                        negative_records.append({
                            'user_id': int(user_id),
                            'session_id': int(pos_record['session_id']),
                            'click_article_id': int(negative_news_id),
                            'click_timestamp': int(click_time),
                            'label': 0
                        })
                        found = True
                
                if len(negative_records) >= n_negative_needed:
                    break
        
        # Add all records from this window
        all_records.extend(positive_records)
        all_records.extend(negative_records[:n_negative_needed])
    
    # Create balanced dataframe and shuffle
    balanced_df = pd.DataFrame(all_records)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save results
    balanced_df.to_csv(output_path, index=False)
    
    print(f"Saved {len(balanced_df):,} samples to {output_path}")
    print(f"  Positive: {len(balanced_df[balanced_df['label']==1]):,}")
    print(f"  Negative: {len(balanced_df[balanced_df['label']==0]):,}")
    
    return balanced_df

def main():
    parser = argparse.ArgumentParser(description='Create balanced windows with negative samples')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--metadata', type=str, default='articles_metadata.csv',
                       help='Path to article metadata')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output CSV file')
    parser.add_argument('--window_size_hours', type=int, default=3,
                       help='Size of time window in hours')
    parser.add_argument('--max_news_age_days', type=int, default=4,
                       help='Maximum age of news articles in days')
    parser.add_argument('--window_length', type=int, default=100,
                       help='Maximum samples per window')
    
    args = parser.parse_args()
    
    create_balanced_windows(
        input_path=args.input,
        metadata_path=args.metadata,
        output_path=args.output,
        window_size_hours=args.window_size_hours,
        max_news_age_days=args.max_news_age_days,
        window_length=args.window_length
    )

if __name__ == '__main__':
    main()