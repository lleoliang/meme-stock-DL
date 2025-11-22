"""
Configuration file for Stream B (Social Encoder) - Stocktwits focused
"""
import os

class Config:
    # Data paths (relative to project root)
    DATA_DIR = "data"
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    MODELS_DIR = "src/models"
    RESULTS_DIR = "results"
    
    # Stocktwits API Configuration
    # Note: Stocktwits has paused new API registrations
    # Options: 'stocktwits' (direct), 'rapidapi' (via RapidAPI), 'pytwits' (community wrapper)
    STOCKTWITS_API_SOURCE = os.environ.get('STOCKTWITS_API_SOURCE', 'stocktwits')  # 'stocktwits', 'rapidapi', 'pytwits'
    
    # Direct Stocktwits API (paused for new users)
    STOCKTWITS_BASE_URL = "https://api.stocktwits.com/api/2"
    STOCKTWITS_ACCESS_TOKEN = os.environ.get('STOCKTWITS_TOKEN', '')  # Set via environment variable
    
    # RapidAPI Configuration (Alternative)
    RAPIDAPI_KEY = os.environ.get('RAPIDAPI_KEY', '')  # Get from https://rapidapi.com/
    RAPIDAPI_STOCKTWITS_URL = "https://stocktwits-api.p.rapidapi.com"
    
    # NLP Sentiment
    USE_NLP_SENTIMENT = True  # Use NLP-based sentiment analysis
    NLP_SENTIMENT_METHOD = 'VADER'  # Options: 'VADER', 'FinBERT'
    
    # Caching
    CACHE_SOCIAL_DATA = True  # Cache API responses
    CACHE_EXPIRY_HOURS = 1  # Cache expiry time
    
    # Model parameters
    SEQUENCE_LENGTH = 60  # T = 60 timesteps
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    DROPOUT = 0.2
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    
    # Social features (Stocktwits)
    SOCIAL_FEATURE_DIM = 3  # [volume, sentiment, velocity]
    
    # Training parameters
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    WEIGHT_DECAY = 1e-5
    
    # Surge definition
    SURGE_THRESHOLD_PCT = 2.0  # 2% price increase
    SURGE_FORWARD_WINDOW = 5  # N sessions forward
    MIN_VOLUME_MULTIPLIER = 1.5  # Volume must be 1.5x average
    
    # Class imbalance
    POSITIVE_CLASS_WEIGHT = 10.0  # Will be tuned during optimization
    FOCAL_LOSS_GAMMA = 2.0
    FOCAL_LOSS_ALPHA = 0.25
    
    # Backtesting
    TOP_K_PREDICTIONS = 10  # Top K stocks to consider
    INITIAL_CAPITAL = 100000  # $100k for backtest
    TRANSACTION_COST_PCT = 0.001  # 0.1% transaction cost
    
    # Data collection
    TICKER_LIMIT = 200  # Top 200 tickers
    MIN_SOCIAL_DATA_POINTS = 30  # Minimum data points required
    
    # Device
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # Random seed
    RANDOM_SEED = 42

