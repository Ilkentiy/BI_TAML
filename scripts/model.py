import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import time
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# ============================================================================
# 1. АРХИТЕКТУРА ДЛЯ NEWSCOLD
# ============================================================================

class NewsColdModel(nn.Module):
    """Модель для новостного холодного старта"""
    def __init__(self, embedding_dim=250, temporal_dim=5):
        super().__init__()
        
        # Размерности
        news_emb_dim = 128
        temp_dim = 64
        time_emb_dim = 16
        
        total_input_dim = news_emb_dim + temp_dim + time_emb_dim
        
        # 1. Энкодер новостей
        self.news_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, news_emb_dim),
            nn.LayerNorm(news_emb_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # 2. Энкодер временных признаков
        self.temporal_encoder = nn.Sequential(
            nn.Linear(temporal_dim, 32),
            nn.GELU(),
            nn.Linear(32, temp_dim),
            nn.LayerNorm(temp_dim),
            nn.GELU()
        )
        
        # 3. Временные эмбеддинги
        self.time_embedding = nn.Embedding(8, time_emb_dim)
        
        # 4. Предсказательный модуль
        self.predictor = nn.Sequential(
            nn.Linear(total_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        self.total_input_dim = total_input_dim
        
    def forward(self, news_emb, temporal_feat, time_period):
        # Кодирование новости
        news_repr = self.news_encoder(news_emb)
        
        # Кодирование временных признаков
        temp_repr = self.temporal_encoder(temporal_feat)
        
        # Временные эмбеддинги
        time_emb = self.time_embedding(time_period.squeeze().long())
        
        # Объединение
        combined = torch.cat([news_repr, temp_repr, time_emb], dim=-1)
        
        # Предсказание
        output = self.predictor(combined)
        return torch.sigmoid(output)