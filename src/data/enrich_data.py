import pandas as pd
import numpy as np
import glob
import os
import argparse

def enrich_with_user_features(df, clicks_folder='data/raw/clicks'):
    """
    Add user context features from original clicks files:
    environment, deviceGroup, os, country, region, referrer_type
    """
    all_clicks = []
    for f in glob.glob(os.path.join(clicks_folder, '*.csv')):
        chunk = pd.read_csv(f, usecols=['user_id', 'click_environment', 
                                        'click_deviceGroup', 'click_os',
                                        'click_country', 'click_region', 
                                        'click_referrer_type'])
        all_clicks.append(chunk)
    
    if not all_clicks:
        print(f"Warning: No files found in {clicks_folder}")
        return df
    
    context_df = pd.concat(all_clicks, ignore_index=True)
    context_df = context_df.drop_duplicates('user_id', keep='first')
    df = df.merge(context_df, on='user_id', how='left')
    return df

def enrich_with_article_metadata(df, metadata_path='data/raw/articles_metadata.csv'):
    """Add article category and creation timestamp"""
    metadata = pd.read_csv(metadata_path)
    metadata = metadata[['article_id', 'category_id', 'created_at_ts']]
    metadata = metadata.rename(columns={'category_id': 'category'})
    
    df = df.merge(metadata, left_on='click_article_id', right_on='article_id', how='left')
    df = df.drop('article_id', axis=1)
    return df

def add_temporal_features(df):
    """
    Add temporal features:
    - time_specific_bin: 3-hour time windows (0-7)
    - time_shift_bin: article age bins (0-5)
    """
    # Convert from milliseconds (13 digits) to datetime
    df['click_timestamp'] = pd.to_datetime(df['click_timestamp'], unit='ms')
    
    # Time-specific bins (3-hour windows)
    df['time_specific_bin'] = (df['click_timestamp'].dt.hour // 3).astype(int)
    
    # Created timestamp from metadata (already in milliseconds)
    df['created_at_ts'] = pd.to_datetime(df['created_at_ts'], unit='ms')
    
    # Age in hours
    df['news_age_hours'] = (df['click_timestamp'] - df['created_at_ts']).dt.total_seconds() / 3600
    
    # Fill NaN with 0
    df['news_age_hours'] = df['news_age_hours'].fillna(0)
    
    # Time-shift bins: 0-1h, 1-3h, 3-6h, 6-12h, 12-24h, >24h
    bins = [0, 1, 3, 6, 12, 24, float('inf')]
    labels = [0, 1, 2, 3, 4, 5]
    
    df['time_shift_bin'] = pd.cut(df['news_age_hours'], bins=bins, labels=labels)
    df['time_shift_bin'] = df['time_shift_bin'].cat.codes
    df['time_shift_bin'] = df['time_shift_bin'].replace(-1, 0)
    
    return df

def enrich_data(input_path, output_path, clicks_folder='data/raw/clicks', metadata_path='data/raw/articles_metadata.csv'):
    """Main function: load, enrich and save data"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = pd.read_csv(input_path)
    print(f"Loaded: {input_path} ({len(df):,} rows)")
    
    df = enrich_with_user_features(df, clicks_folder)
    df = enrich_with_article_metadata(df, metadata_path)
    df = add_temporal_features(df)
    
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({len(df):,} rows)")
    return df

def main():
    parser = argparse.ArgumentParser(description='Enrich data with features')
    parser.add_argument('--input', type=str, 
                       default='data/processed/split_positive_negative/train.csv',
                       help='Input CSV path')
    parser.add_argument('--output', type=str,
                       default='data/processed/enriched/train_enriched.csv',
                       help='Output CSV path')
    parser.add_argument('--clicks_folder', type=str, default='data/raw/clicks',
                       help='Clicks folder path')
    parser.add_argument('--metadata', type=str, default='data/raw/articles_metadata.csv',
                       help='Metadata path')
    
    args = parser.parse_args()
    enrich_data(args.input, args.output, args.clicks_folder, args.metadata)

if __name__ == '__main__':
    main()