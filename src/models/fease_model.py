import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset


class FEASEDataset(Dataset):
    def __init__(self, df, embeddings_path='data/raw/articles_embeddings.pickle'):
        with open(embeddings_path, 'rb') as f:
            embeddings_data = pickle.load(f)
        
        self.article_embeddings = {}
        for i, emb in enumerate(embeddings_data):
            self.article_embeddings[i] = emb.astype(np.float32)
        
        print(f"Loaded {len(self.article_embeddings):,} article embeddings")
        
        self.df = df.copy()
        
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
        ]
        
        time_specific = safe_get('time_specific_bin', 8)
        
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
            'time_specific': torch.LongTensor([time_specific]),
            'content_emb': torch.FloatTensor(content_emb),
            'category': torch.LongTensor([category]),
            'time_shift': torch.LongTensor([time_shift]),
            'news_age': torch.FloatTensor([news_age]),
            'label': torch.FloatTensor([label])
        }


class FeatureSelection(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2)
        
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels * 2)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        
        pooled = torch.mean(x, dim=2, keepdim=True).squeeze(-1)
        
        attn = self.fc1(pooled)
        attn = F.relu(attn)
        attn = self.fc2(attn)
        attn = attn.view(attn.size(0), 2, -1)
        attn = F.softmax(attn, dim=1)
        
        a1 = attn[:, 0, :].unsqueeze(-1)
        a2 = attn[:, 1, :].unsqueeze(-1)
        
        return a1 * x1 + a2 * x2


class FeatureEnhancement(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        
    def forward(self, x):
        pooled = torch.mean(x, dim=2, keepdim=True).squeeze(-1)
        
        attn = self.fc1(pooled)
        attn = F.relu(attn)
        attn = self.fc2(attn)
        attn = torch.sigmoid(attn).unsqueeze(-1)
        
        return x * attn


class FEASEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.fs = FeatureSelection(in_channels, reduction)
        self.fe = FeatureEnhancement(in_channels, reduction)
        
    def forward(self, x):
        x = self.fs(x)
        x = self.fe(x)
        return x


class FEASENewsRanker(nn.Module):
    def __init__(self, emb_sizes, context_dim=6, time_dim=8, 
                 content_dim=250, category_dim=461, time_shift_dim=6,
                 hidden_dim=256, reduction=16, dropout=0.3):
        super().__init__()
        
        self.context_emb = nn.Embedding(emb_sizes['click_environment'], 8)
        self.device_emb = nn.Embedding(emb_sizes['click_deviceGroup'], 8)
        self.os_emb = nn.Embedding(emb_sizes['click_os'], 8)
        self.country_emb = nn.Embedding(emb_sizes['click_country'], 8)
        self.region_emb = nn.Embedding(emb_sizes['click_region'], 8)
        self.referrer_emb = nn.Embedding(emb_sizes['click_referrer_type'], 8)
        self.time_specific_emb = nn.Embedding(emb_sizes['time_specific_bin'], 8)
        self.category_emb = nn.Embedding(emb_sizes['category'], 16)
        self.time_shift_emb = nn.Embedding(emb_sizes['time_shift_bin'], 8)
        
        user_dim = 6 * 8 + 8
        item_dim = content_dim + 16 + 8 + 1
        
        self.user_fease = FEASEModule(user_dim, reduction)
        self.item_fease = FEASEModule(item_dim, reduction)
        
        self.user_encoder = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 64)
        )
        
        self.item_encoder = nn.Sequential(
            nn.Linear(item_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 64)
        )
        
        self.scorer = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, context, time_specific, content_emb, category, time_shift, news_age):
        ctx_emb = torch.cat([
            self.context_emb(context[:, 0]),
            self.device_emb(context[:, 1]),
            self.os_emb(context[:, 2]),
            self.country_emb(context[:, 3]),
            self.region_emb(context[:, 4]),
            self.referrer_emb(context[:, 5]),
            self.time_specific_emb(time_specific.squeeze()),
        ], dim=1)
        
        user_features = ctx_emb
        
        cat_emb = self.category_emb(category.squeeze())
        ts_emb = self.time_shift_emb(time_shift.squeeze())
        
        item_features = torch.cat([content_emb, cat_emb, ts_emb, news_age], dim=1)
        
        user_features = self.user_fease(user_features.unsqueeze(-1)).squeeze(-1)
        item_features = self.item_fease(item_features.unsqueeze(-1)).squeeze(-1)
        
        user_vec = self.user_encoder(user_features)
        item_vec = self.item_encoder(item_features)
        
        user_norm = F.normalize(user_vec, p=2, dim=1)
        item_norm = F.normalize(item_vec, p=2, dim=1)
        score = torch.sum(user_norm * item_norm, dim=1, keepdim=True)
        
        return self.scorer(score)
    
    def get_user_vector(self, context, time_specific):
        ctx_emb = torch.cat([
            self.context_emb(context[:, 0]),
            self.device_emb(context[:, 1]),
            self.os_emb(context[:, 2]),
            self.country_emb(context[:, 3]),
            self.region_emb(context[:, 4]),
            self.referrer_emb(context[:, 5]),
            self.time_specific_emb(time_specific.squeeze()),
        ], dim=1)
        
        user_features = self.user_fease(ctx_emb.unsqueeze(-1)).squeeze(-1)
        user_vec = self.user_encoder(user_features)
        return F.normalize(user_vec, p=2, dim=1)
    
    def get_item_vector(self, content_emb, category, time_shift, news_age):
        cat_emb = self.category_emb(category.squeeze())
        ts_emb = self.time_shift_emb(time_shift.squeeze())
        
        item_features = torch.cat([content_emb, cat_emb, ts_emb, news_age], dim=1)
        item_features = self.item_fease(item_features.unsqueeze(-1)).squeeze(-1)
        item_vec = self.item_encoder(item_features)
        return F.normalize(item_vec, p=2, dim=1)