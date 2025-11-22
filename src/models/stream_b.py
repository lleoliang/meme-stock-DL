"""
Stream B: Social Encoder Model
Processes Stocktwits social signals (volume, sentiment, velocity)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config

class SocialEncoder(nn.Module):
    """
    Stream B: Encoder for social signals
    Can use LSTM, GRU, or Transformer
    """
    
    def __init__(self, 
                 input_dim: int = Config.SOCIAL_FEATURE_DIM,
                 hidden_dim: int = Config.HIDDEN_DIM,
                 num_layers: int = Config.NUM_LAYERS,
                 dropout: float = Config.DROPOUT,
                 encoder_type: str = 'LSTM'):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder_type = encoder_type
        
        if encoder_type == 'LSTM':
            self.encoder = nn.LSTM(
                input_dim, 
                hidden_dim, 
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif encoder_type == 'GRU':
            self.encoder = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif encoder_type == 'Transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            # Project input to hidden_dim
            self.input_projection = nn.Linear(input_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [B, T, input_dim] social sequence
        Returns:
            output: [B, T, hidden_dim] encoded sequence
        """
        if self.encoder_type == 'Transformer':
            x = self.input_projection(x)  # [B, T, hidden_dim]
            output = self.encoder(x)
        else:
            output, _ = self.encoder(x)  # [B, T, hidden_dim]
        
        output = self.dropout(output)
        return output


class StreamBClassifier(nn.Module):
    """
    Complete Stream B model: Social Encoder + Classifier
    Predicts surge probability from social signals alone
    """
    
    def __init__(self,
                 input_dim: int = Config.SOCIAL_FEATURE_DIM,
                 hidden_dim: int = Config.HIDDEN_DIM,
                 num_layers: int = Config.NUM_LAYERS,
                 dropout: float = Config.DROPOUT,
                 encoder_type: str = 'LSTM',
                 use_attention: bool = True):
        super().__init__()
        
        self.social_encoder = SocialEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            encoder_type=encoder_type
        )
        
        self.use_attention = use_attention
        
        if use_attention:
            # Self-attention pooling
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, social_seq):
        """
        Args:
            social_seq: [B, T, input_dim] social features
        Returns:
            logits: [B, 1] binary classification logits
        """
        # Encode social sequence
        encoded = self.social_encoder(social_seq)  # [B, T, hidden_dim]
        
        # Pooling: attention or mean
        if self.use_attention:
            # Self-attention pooling
            attn_out, attn_weights = self.attention(encoded, encoded, encoded)
            attn_out = self.attention_norm(attn_out + encoded)  # Residual
            pooled = attn_out.mean(dim=1)  # [B, hidden_dim]
        else:
            # Simple mean pooling
            pooled = encoded.mean(dim=1)  # [B, hidden_dim]
        
        # Classify
        logits = self.classifier(pooled)  # [B, 1]
        
        return logits.squeeze(-1)  # [B]

