import torch
import torch.nn as nn
import numpy as np

class MemeStockModel(nn.Module):
    """
    Two-stream model: Market encoder + Social encoder + Cross-attention + Classifier
    
    Mathematical notation:
        M_t ∈ ℝ^(T × d_m) - Market sequence
        S_t ∈ ℝ^(T × d_s) - Social sequence  
        p_t ∈ ℝ^(d_p)     - Parabolic features
        y_t ∈ {0, 1}      - Binary label
    """
    
    def __init__(
        self,
        T=14,           # Sequence length (window days)
        d_m=5,          # Market features (OHLCV)
        d_s=3,          # Social features (for now, placeholder)
        d_p=4,          # Parabolic features (a, h, k, mse)
        h=64,           # Hidden dimension
    ):
        super().__init__()
        
        self.T = T
        self.d_m = d_m
        self.d_s = d_s
        self.d_p = d_p
        self.h = h
        
        self.market_encoder = nn.LSTM(
            input_size=d_m,
            hidden_size=h,
            num_layers=1,
            batch_first=True
        )
        
        self.social_encoder = nn.LSTM(
            input_size=d_s,
            hidden_size=h,
            num_layers=1,
            batch_first=True
        )
        
        # Cross-Attention: CA_φ(H_t^(m), H_t^(s)) → H̃_t^(m)
        # Market attends to Social
        # This is simplified - no multi-head for now
        self.W_Q = nn.Linear(h, h)  # Query projection
        self.W_K = nn.Linear(h, h)  # Key projection
        self.W_V = nn.Linear(h, h)  # Value projection
        
        # Classifier: h_ψ([z_m, z_s, p_t]) → u_t → ŷ_t
        # Input: [z_m, z_s, p_t] ∈ ℝ^(2h + d_p)
        # Output: u_t ∈ ℝ (logit)
        self.classifier = nn.Sequential(
            nn.Linear(2*h + d_p, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, M_t, S_t, p_t):
        """
        Forward pass: (M_t, S_t, p_t) → ŷ_t
        
        Args:
            M_t: Market sequence, shape (batch, T, d_m)
            S_t: Social sequence, shape (batch, T, d_s)
            p_t: Parabolic features, shape (batch, d_p)
        
        Returns:
            logits: shape (batch, 1) - raw scores (before sigmoid)
        """

        # Step 1: Encode Market
        # H_t^(m) = f_θm(M_t)
        H_m, _ = self.market_encoder(M_t)  # (batch, T, h)
        
        # Step 2: Encode Social
        # H_t^(s) = g_θs(S_t)
        H_s, _ = self.social_encoder(S_t)  # (batch, T, h)
        
        # Step 3: Cross-Attention
        # H̃_t^(m) = CA_φ(H_t^(m), H_t^(s))
        # Market queries, Social provides keys/values
        
        Q = self.W_Q(H_m)  # (batch, T, h) - Queries from market
        K = self.W_K(H_s)  # (batch, T, h) - Keys from social
        V = self.W_V(H_s)  # (batch, T, h) - Values from social
        
        # Attention scores: Q @ K^T / sqrt(h)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.h)  # (batch, T, T)
        attention_weights = torch.softmax(scores, dim=-1)  # (batch, T, T)
        
        # Attended market: A @ V
        H_m_attended = torch.matmul(attention_weights, V)  # (batch, T, h)
        
        # Step 4: Pooling (sequence → vector)
        # z_m = mean(H̃_t^(m))
        # z_s = mean(H_t^(s))
        z_m = H_m_attended.mean(dim=1)  # (batch, h)
        z_s = H_s.mean(dim=1)            # (batch, h)
        
        # Step 5: Concatenate with parabolic features
        # z_t = [z_m, z_s, p_t]
        z_t = torch.cat([z_m, z_s, p_t], dim=1)  # (batch, 2h + d_p)
        
        # Step 6: Classifier
        # u_t = h_ψ(z_t)
        logits = self.classifier(z_t)  # (batch, 1)
        
        return logits
    
    def predict_probability(self, M_t, S_t, p_t):
        """
        Get probability predictions (apply sigmoid)
        
        Returns:
            probabilities: shape (batch, 1) - values in [0, 1]
        """
        logits = self.forward(M_t, S_t, p_t)
        probs = torch.sigmoid(logits)
        return probs


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    print("=== Testing Meme Stock Model ===\n")
    
    # Model hyperparameters
    T = 14      # 60-day window
    d_m = 5     # OHLCV features
    d_s = 3     # Social features (placeholder)
    d_p = 4     # Parabolic features
    h = 64      # Hidden dimension
    
    # Create model
    model = MemeStockModel(T=T, d_m=d_m, d_s=d_s, d_p=d_p, h=h)
    
    print(f"Model created with:")
    print(f"  Sequence length T = {T}")
    print(f"  Market features d_m = {d_m}")
    print(f"  Social features d_s = {d_s}")
    print(f"  Hidden dimension h = {h}")
    print(f"  Parabolic features d_p = {d_p}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # ============================================
    # Create dummy data for ONE training example
    # ============================================
    batch_size = 1
    
    # M_t: Market sequence (OHLCV over 60 days)
    M_t = torch.randn(batch_size, T, d_m)
    print(f"\nM_t shape: {M_t.shape}  (batch, T, d_m)")
    
    # S_t: Social sequence (placeholder - normally from Reddit)
    S_t = torch.randn(batch_size, T, d_s)
    print(f"S_t shape: {S_t.shape}  (batch, T, d_s)")
    
    # p_t: Parabolic features (a, h, k, mse)
    p_t = torch.randn(batch_size, d_p)
    print(f"p_t shape: {p_t.shape}  (batch, d_p)")
    
    # ============================================
    # Forward pass
    # ============================================
    print("\n--- Running forward pass ---")
    
    logits = model(M_t, S_t, p_t)
    probs = model.predict_probability(M_t, S_t, p_t)
    
    print(f"\nOutput logit: {logits.item():.4f}")
    print(f"Output probability: {probs.item():.4f}")
    
    # Interpret
    if probs.item() > 0.5:
        print("→ Model predicts: SURGE (y_t = 1)")
    else:
        print("→ Model predicts: NO SURGE (y_t = 0)")
    
    print("\n=== Model test successful! ===")