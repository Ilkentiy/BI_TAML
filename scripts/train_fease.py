import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import json
import argparse
from datetime import datetime

from scripts.config import *
from src.models.fease_model import FEASENewsRanker, FEASEDataset


def train(args):
    print("Training FEASE baseline")
    print(f"Device: {DEVICE}")
    
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    print(f"Train: {len(train_df):,} rows, Test: {len(test_df):,} rows")
    
    train_data, val_data = train_test_split(
        train_df, 
        test_size=args.test_size, 
        random_state=args.random_state,
        stratify=train_df['label']
    )
    print(f"Train split: {len(train_data):,}, Val split: {len(val_data):,}")
    
    train_dataset = FEASEDataset(train_data, embeddings_path=args.embeddings_path)
    val_dataset = FEASEDataset(val_data, embeddings_path=args.embeddings_path)
    test_dataset = FEASEDataset(test_df, embeddings_path=args.embeddings_path)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    emb_sizes = train_dataset.emb_sizes
    model = FEASENewsRanker(emb_sizes=emb_sizes, **FEASE_CONFIG)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    model, history = train_model(model, train_loader, val_loader, {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'early_stopping_patience': args.early_stopping_patience,
        'model_save_path': args.model_save_path
    })
    
    results = evaluate_model(model, test_loader)
    
    results_file = os.path.join(
        args.results_dir, 
        f'fease_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    with open(results_file, 'w') as f:
        json.dump({
            'model': 'FEASE',
            'best_val_auc': max(history['val_auc']),
            'test_auc': results['auc'],
            'total_params': total_params,
            'history': history
        }, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    return model, history, results


def train_model(model, train_loader, val_loader, config):
    model = model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = torch.nn.BCELoss()
    
    best_val_auc = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': []}
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch in train_loader:
            context = batch['context'].to(DEVICE)
            time_specific = batch['time_specific'].to(DEVICE)
            content_emb = batch['content_emb'].to(DEVICE)
            category = batch['category'].to(DEVICE)
            time_shift = batch['time_shift'].to(DEVICE)
            news_age = batch['news_age'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(context, time_specific, content_emb, category, time_shift, news_age)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(preds.detach().cpu().numpy().flatten())
            train_labels.extend(labels.cpu().numpy().flatten())
        
        avg_train_loss = train_loss / len(train_loader)
        train_auc = roc_auc_score(train_labels, train_preds)
        
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                context = batch['context'].to(DEVICE)
                time_specific = batch['time_specific'].to(DEVICE)
                content_emb = batch['content_emb'].to(DEVICE)
                category = batch['category'].to(DEVICE)
                time_shift = batch['time_shift'].to(DEVICE)
                news_age = batch['news_age'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                
                preds = model(context, time_specific, content_emb, category, time_shift, news_age)
                loss = criterion(preds, labels)
                
                val_loss += loss.item()
                val_preds.extend(preds.cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy().flatten())
        
        avg_val_loss = val_loss / len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)
        
        history['train_loss'].append(avg_train_loss)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(val_auc)
        
        scheduler.step(val_auc)
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f}, AUC: {train_auc:.4f} | Val Loss: {avg_val_loss:.4f}, AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), config['model_save_path'])
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config['early_stopping_patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load(config['model_save_path']))
    return model, history


def evaluate_model(model, test_loader):
    model.eval()
    preds = []
    labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            context = batch['context'].to(DEVICE)
            time_specific = batch['time_specific'].to(DEVICE)
            content_emb = batch['content_emb'].to(DEVICE)
            category = batch['category'].to(DEVICE)
            time_shift = batch['time_shift'].to(DEVICE)
            news_age = batch['news_age'].to(DEVICE)
            
            outputs = model(context, time_specific, content_emb, category, time_shift, news_age)
            preds.extend(outputs.cpu().numpy().flatten())
            labels.extend(batch['label'].numpy().flatten())
    
    auc = roc_auc_score(labels, preds)
    print(f"Test AUC: {auc:.4f}")
    return {'auc': auc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FEASE baseline')
    parser.add_argument('--train_path', type=str, default=TRAIN_PATH)
    parser.add_argument('--test_path', type=str, default=TEST_PATH)
    parser.add_argument('--embeddings_path', type=str, default=EMBEDDINGS_PATH)
    parser.add_argument('--model_save_path', type=str, default=os.path.join(MODELS_DIR, 'fease_ranker.pth'))
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--early_stopping_patience', type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument('--test_size', type=float, default=TEST_SIZE)
    parser.add_argument('--random_state', type=int, default=RANDOM_STATE)
    
    args = parser.parse_args()
    train(args)