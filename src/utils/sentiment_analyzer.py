"""
NLP-based sentiment analysis for Stocktwits messages
Supports VADER and FinBERT
"""
import os
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    """NLP-based sentiment analyzer"""
    
    def __init__(self, method: str = 'VADER'):
        """
        Initialize sentiment analyzer
        
        Args:
            method: 'VADER' (fast, no GPU) or 'FinBERT' (accurate, requires GPU/transformers)
        """
        self.method = method
        self.analyzer = None
        self._initialize_analyzer()
    
    def _initialize_analyzer(self):
        """Initialize the chosen sentiment analyzer"""
        if self.method == 'VADER':
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self.analyzer = SentimentIntensityAnalyzer()
                print("VADER sentiment analyzer initialized")
            except ImportError:
                print("VADER not installed. Install with: pip install vaderSentiment")
                self.analyzer = None
        
        elif self.method == 'FinBERT':
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                
                model_name = "ProsusAI/finbert"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.eval()
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
                print(f"FinBERT sentiment analyzer initialized on {self.device}")
            except ImportError:
                print("Transformers not installed. Install with: pip install transformers")
                self.analyzer = None
        else:
            raise ValueError(f"Unknown sentiment method: {self.method}")
    
    def analyze(self, text: str) -> float:
        """
        Analyze sentiment of text
        
        Returns:
            Sentiment score from -1.0 (very negative) to 1.0 (very positive)
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        if self.method == 'VADER':
            if self.analyzer is None:
                return 0.0
            scores = self.analyzer.polarity_scores(text)
            # Compound score ranges from -1 to 1
            return scores['compound']
        
        elif self.method == 'FinBERT':
            if not hasattr(self, 'model'):
                return 0.0
            
            import torch
            
            # Tokenize and predict
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT outputs: [positive, negative, neutral]
            # Map to -1 to 1 scale: positive - negative
            sentiment_score = (probs[0][0] - probs[0][1]).item()
            return sentiment_score
        
        return 0.0
    
    def batch_analyze(self, texts: list) -> list:
        """Analyze multiple texts efficiently"""
        return [self.analyze(text) for text in texts]

