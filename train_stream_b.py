"""
Training script for Stream B with weight optimization
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import json
from datetime import datetime

from models.stream_b import StreamBClassifier
from losses import FocalLoss, WeightedBCELoss
from config import Config

class SocialDataset(Dataset):
    """Dataset for social sequences"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class StreamBTrainer:
    """Trainer for Stream B model with hyperparameter optimization"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.val_pr_aucs = []
        
    def train_epoch(self, dataloader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            logits = self.model(X)
            loss = criterion(logits, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader, criterion):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss = criterion(logits, y)
                total_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate PR-AUC
        precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
        pr_auc = auc(recall, precision)
        
        # Calculate precision@K (top 10% predictions)
        k = max(1, int(len(all_preds) * 0.1))
        top_k_indices = np.argsort(all_preds)[-k:]
        precision_at_k = np.mean(all_labels[top_k_indices])
        
        return {
            'loss': total_loss / len(dataloader),
            'pr_auc': pr_auc,
            'precision_at_k': precision_at_k,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def train(self, train_loader, val_loader, 
              criterion, optimizer, num_epochs, 
              early_stopping_patience=10):
        """Train with early stopping"""
        best_pr_auc = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            val_metrics = self.evaluate(val_loader, criterion)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.val_pr_aucs.append(val_metrics['pr_auc'])
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val PR-AUC: {val_metrics['pr_auc']:.4f}")
            print(f"  Precision@K: {val_metrics['precision_at_k']:.4f}")
            
            # Early stopping
            if val_metrics['pr_auc'] > best_pr_auc:
                best_pr_auc = val_metrics['pr_auc']
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return best_pr_auc


def optimize_weights(train_data, val_data, 
                     weight_range=[1.0, 5.0, 10.0, 20.0, 50.0],
                     use_focal_loss=False):
    """
    Optimize class weights or focal loss parameters
    """
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = SocialDataset(train_data['X'], train_data['y'])
    val_dataset = SocialDataset(val_data['X'], val_data['y'])
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    results = []
    
    for weight in weight_range:
        print(f"\n{'='*50}")
        print(f"Testing weight: {weight}")
        print(f"{'='*50}")
        
        # Create model
        model = StreamBClassifier(
            input_dim=Config.SOCIAL_FEATURE_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            num_layers=Config.NUM_LAYERS,
            dropout=Config.DROPOUT,
            encoder_type='LSTM',
            use_attention=True
        )
        
        # Create loss
        if use_focal_loss:
            criterion = FocalLoss(alpha=weight/100.0, gamma=Config.FOCAL_LOSS_GAMMA)
        else:
            criterion = WeightedBCELoss(pos_weight=weight)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Train
        trainer = StreamBTrainer(model, device=device)
        best_pr_auc = trainer.train(
            train_loader, val_loader,
            criterion, optimizer,
            num_epochs=Config.NUM_EPOCHS,
            early_stopping_patience=Config.EARLY_STOPPING_PATIENCE
        )
        
        # Final evaluation
        final_metrics = trainer.evaluate(val_loader, criterion)
        
        results.append({
            'weight': weight,
            'pr_auc': best_pr_auc,
            'final_pr_auc': final_metrics['pr_auc'],
            'precision_at_k': final_metrics['precision_at_k']
        })
        
        print(f"Best PR-AUC: {best_pr_auc:.4f}")
        print(f"Final PR-AUC: {final_metrics['pr_auc']:.4f}")
        print(f"Precision@K: {final_metrics['precision_at_k']:.4f}")
    
    # Find best weight
    results_df = pd.DataFrame(results)
    best_result = results_df.loc[results_df['pr_auc'].idxmax()]
    
    print(f"\n{'='*50}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*50}")
    print(results_df)
    print(f"\nBest weight: {best_result['weight']}")
    print(f"Best PR-AUC: {best_result['pr_auc']:.4f}")
    
    return best_result, results_df


if __name__ == "__main__":
    # This will be called from main script after data preparation
    print("Stream B Training Module")
    print("Use prepare_and_train.py to run full pipeline")

