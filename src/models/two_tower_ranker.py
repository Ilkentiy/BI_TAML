import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset

class TwoTowerDataset(Dataset):
    def __init__(self, df, embeddings_path='data/raw/articles_embeddings.pickle', 
                 cluster_path='data/processed/enriched/user_clusters.pkl'):
        
        with open(embeddings_path, 'rb') as f:
            embeddings_data = pickle.load(f)
        
        self.article_embeddings = {}
        for i, emb in enumerate(embeddings_data):
            self.article_embeddings[i] = emb.astype(np.float32)
        
        print(f"Loaded {len(self.article_embeddings):,} article embeddings")
        
        with open(cluster_path, 'rb') as f:
            cluster_data = pickle.load(f)
            self.cluster_map = cluster_data['cluster_map']
        
        self.df = df.copy()
        self.df['user_cluster'] = self.df['user_id'].map(self.cluster_map).fillna(0).astype(int)
        
        self.emb_sizes = {
            'click_environment': 5,
            'click_deviceGroup': 6,
            'click_os': 21,
            'click_country': 12,
            'click_region': 29,
            'click_referrer_type': 8,
            'time_specific_bin': 8,
            'time_shift_bin': 6,
            'category': 461,
            'cluster': 51
        }
        
        print("Embedding sizes:")
        for k, v in self.emb_sizes.items():
            print(f"  {k}: {v}")
        
        print(f"Loaded {len(self.df):,} samples")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        def safe_get(col, max_val, default=0):
            val = row.get(col, default)
            if pd.isna(val) or val < 0:
                val = default
            val = int(val)
            if val >= max_val:
                val = max_val - 1
            return val
        
        context = [
            safe_get('click_environment', 5),
            safe_get('click_deviceGroup', 6),
            safe_get('click_os', 21),
            safe_get('click_country', 12),
            safe_get('click_region', 29),
            safe_get('click_referrer_type', 8),
            safe_get('time_specific_bin', 8),
            safe_get('user_cluster', 51),
        ]
        
        article_id = row['click_article_id']
        if isinstance(article_id, (int, float)):
            article_id = int(article_id)
        content_emb = self.article_embeddings.get(article_id)
        if content_emb is None:
            content_emb = np.zeros(250, dtype=np.float32)
        else:
            content_emb = content_emb.astype(np.float32)
        
        category = safe_get('category', 461)
        time_shift = safe_get('time_shift_bin', 6)
        
        news_age = row.get('news_age_hours', 0)
        if pd.isna(news_age):
            news_age = 0
        news_age = min(news_age / 72.0, 1.0)
        
        label = row.get('label', 0)
        if pd.isna(label):
            label = 0
        
        return {
            'context': torch.LongTensor(context),
            'content_emb': torch.FloatTensor(content_emb),
            'category': torch.LongTensor([category]),
            'time_shift_bin': torch.LongTensor([time_shift]),
            'news_age': torch.FloatTensor([news_age]),
            'label': torch.FloatTensor([label])
        }


class UserTower(nn.Module):
    def __init__(self, emb_sizes, emb_dim=16, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        
        self.env_emb = nn.Embedding(emb_sizes['click_environment'], emb_dim)
        self.device_emb = nn.Embedding(emb_sizes['click_deviceGroup'], emb_dim)
        self.os_emb = nn.Embedding(emb_sizes['click_os'], emb_dim)
        self.country_emb = nn.Embedding(emb_sizes['click_country'], emb_dim)
        self.region_emb = nn.Embedding(emb_sizes['click_region'], emb_dim)
        self.referrer_emb = nn.Embedding(emb_sizes['click_referrer_type'], emb_dim)
        self.time_specific_emb = nn.Embedding(emb_sizes['time_specific_bin'], emb_dim)
        self.cluster_emb = nn.Embedding(emb_sizes['cluster'], emb_dim)
        
        input_dim = 8 * emb_dim
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 64))
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, context):
        env = self.env_emb(context[:, 0])
        device = self.device_emb(context[:, 1])
        os = self.os_emb(context[:, 2])
        country = self.country_emb(context[:, 3])
        region = self.region_emb(context[:, 4])
        referrer = self.referrer_emb(context[:, 5])
        time_spec = self.time_specific_emb(context[:, 6])
        cluster = self.cluster_emb(context[:, 7])
        
        combined = torch.cat([env, device, os, country, region, 
                             referrer, time_spec, cluster], dim=1)
        
        return self.encoder(combined)


class ItemTower(nn.Module):
    def __init__(self, emb_sizes, content_dim=250, category_emb_dim=32, time_shift_emb_dim=16,
                 hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        
        self.category_emb = nn.Embedding(emb_sizes['category'], category_emb_dim)
        self.time_shift_emb = nn.Embedding(emb_sizes['time_shift_bin'], time_shift_emb_dim)
        
        input_dim = content_dim + category_emb_dim + time_shift_emb_dim + 1
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 64))
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, content_emb, category, time_shift_bin, news_age):
        cat_emb = self.category_emb(category.squeeze())
        time_shift_emb = self.time_shift_emb(time_shift_bin.squeeze())
        
        combined = torch.cat([content_emb, cat_emb, time_shift_emb, news_age], dim=1)
        
        return self.encoder(combined)


class TwoTowerRanker(nn.Module):
    def __init__(self, config, emb_sizes):
        super().__init__()
        
        self.user_tower = UserTower(emb_sizes, **config.get('user_tower', {}))
        self.item_tower = ItemTower(emb_sizes, **config.get('item_tower', {}))
        
        self.scorer = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, context, content_emb, category, time_shift_bin, news_age):
        user_vec = self.user_tower(context)
        item_vec = self.item_tower(content_emb, category, time_shift_bin, news_age)
        
        user_norm = F.normalize(user_vec, p=2, dim=1)
        item_norm = F.normalize(item_vec, p=2, dim=1)
        score = torch.sum(user_norm * item_norm, dim=1, keepdim=True)
        
        return self.scorer(score)
    
    def get_user_vector(self, context):
        user_vec = self.user_tower(context)
        return F.normalize(user_vec, p=2, dim=1)
    
    def get_item_vector(self, content_emb, category, time_shift_bin, news_age):
        item_vec = self.item_tower(content_emb, category, time_shift_bin, news_age)
        return F.normalize(item_vec, p=2, dim=1)