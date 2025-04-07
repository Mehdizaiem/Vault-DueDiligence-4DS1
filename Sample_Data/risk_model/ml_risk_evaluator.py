from typing import Dict, List
from datetime import datetime
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

# Setup path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class MLRiskEvaluator:
    def __init__(self, model_path: str = None, thresholds: Dict[str, float] = None):
        self.history = []
        self.thresholds = thresholds or {
            "regulatory": 0.7,
            "market": 0.6,
            "technical": 0.65,
            "operational": 0.6,
            "fraud": 0.5
        }
        
        # Default model path if none provided
        if not model_path:
            model_path = str(Path(__file__).parent / "models" / "risk_model.pkl")
        
        try:
            self.model = self._load_model(model_path)
            self.scaler = self._load_scaler(str(Path(model_path).parent / "scaler.pkl"))
            logger.info(f"Successfully loaded model from {model_path}")
            self.model_loaded = True
        except Exception as e:
            logger.warning(f"Could not load model: {e}. Falling back to rule-based evaluation.")
            self.model_loaded = False
    
    def _load_model(self, model_path: str):
        """Load the pretrained model from disk"""
        try:
            return joblib.load(model_path)
        except FileNotFoundError:
            logger.warning(f"Model file not found at {model_path}")
            return None
    
    def _load_scaler(self, scaler_path: str):
        """Load the feature scaler from disk"""
        try:
            return joblib.load(scaler_path)
        except FileNotFoundError:
            logger.warning(f"Scaler not found at {scaler_path}, will use default standardization")
            return StandardScaler()
            
    def evaluate(self, fund_id: str, market_data: Dict[str, float]) -> Dict[str, any]:
        """
        Evaluate multi-dimensional risk based on market snapshot using 
        either ML model or fallback to rule-based logic.
        
        Parameters:
        fund_id (str): Identifier for the fund
        market_data (Dict[str, float]): Dictionary containing market metrics
        
        Returns:
        Dict[str, any]: Risk profile with scores for different risk dimensions
        """
        # Prepare input features
        features = self._prepare_features(market_data)
        
        # Initialize risk profile with basic info
        risk_profile = {
            "fund_id": fund_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",  # ✅ Fix
        }
        
        # Use ML model if available, otherwise fall back to rule-based
        if self.model_loaded and self.model is not None:
            try:
                # Get predictions from model for each risk dimension
                risk_scores = self._predict_risk_scores(features)
                
                # Add predictions to risk profile
                risk_profile.update(risk_scores)
                logger.info(f"Generated ML-based risk profile for {fund_id}")
            except Exception as e:
                logger.error(f"Error using ML model: {e}. Falling back to rule-based.")
                risk_profile.update(self._rule_based_evaluation(market_data))
        else:
            # Use rule-based approach
            risk_profile.update(self._rule_based_evaluation(market_data))
            
        # Calculate overall score from risk dimensions
        risk_factors = ["regulatory", "market", "technical", "operational", "fraud"]
        risk_values = [risk_profile[factor] for factor in risk_factors]
        risk_profile["overall_score"] = round(sum(risk_values) / len(risk_values), 3)
        
        # Generate alerts based on thresholds
        alerts = [
            f"{k.upper()} RISK: {v}" 
            for k, v in risk_profile.items() 
            if isinstance(v, float) and k in self.thresholds and v >= self.thresholds[k]
        ]
        risk_profile["alerts"] = alerts
        
        # Add to history
        self.history.append(risk_profile)
        
        logger.info(f"[RiskEvaluator] Profiled fund {fund_id}: overall_score={risk_profile['overall_score']}")
        return risk_profile
    
    def _prepare_features(self, market_data: Dict[str, float]) -> np.ndarray:
        """Convert market data to feature vector for model input"""
        # Define expected features in the correct order for the model
        expected_features = ['volume_24h', 'price_change_24h', 'market_cap', 
                             'volatility', 'liquidity_ratio']
        
        # Extract features, using 0 for missing ones
        feature_dict = {feat: market_data.get(feat, 0) for feat in expected_features}
        
        # Add derived features that might help the model
        feature_dict['price_volatility'] = abs(feature_dict['price_change_24h'])
        feature_dict['log_market_cap'] = np.log1p(feature_dict['market_cap']) if feature_dict['market_cap'] > 0 else 0
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame([feature_dict])
        
        # Scale features
        if self.scaler:
            try:
                return self.scaler.transform(df)
            except:
                # If scaler fails (e.g., different feature set), fit a new one
                return StandardScaler().fit_transform(df)
        else:
            return df.values
    
    def _predict_risk_scores(self, features: np.ndarray) -> Dict[str, float]:
        """Get risk scores from model predictions"""
        # Model should output risk scores for each dimension
        # This implementation assumes the model returns 5 values
        try:
            predictions = self.model.predict(features)[0]
            
            # Ensure predictions are within [0,1] range
            predictions = np.clip(predictions, 0, 1)
            
            # Map to risk dimensions
            risk_dimensions = ["regulatory", "market", "technical", "operational", "fraud"]
            
            # If model returns single value, distribute with some variation
            if len(predictions) == 1:
                base_value = predictions[0]
                predictions = [
                    min(1.0, max(0.0, base_value + np.random.normal(0, 0.05)))
                    for _ in range(len(risk_dimensions))
                ]
            
            # If model returns different number of predictions than expected
            if len(predictions) != len(risk_dimensions):
                logger.warning(f"Model returned {len(predictions)} values, expected {len(risk_dimensions)}")
                # Pad or truncate to match expected dimensions
                if len(predictions) < len(risk_dimensions):
                    predictions = list(predictions) + [0.5] * (len(risk_dimensions) - len(predictions))
                else:
                    predictions = predictions[:len(risk_dimensions)]
            
            return {dim: round(float(pred), 3) for dim, pred in zip(risk_dimensions, predictions)}
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            # Return default values on error
            return {
                "regulatory": 0.5,
                "market": 0.6,
                "technical": 0.5,
                "operational": 0.5,
                "fraud": 0.5
            }
    
    def _rule_based_evaluation(self, market_data: Dict[str, float]) -> Dict[str, float]:
        """Fallback rule-based risk evaluation logic"""
        volume = market_data.get("volume_24h", 0)
        price_change = abs(market_data.get("price_change_24h", 0))
        market_cap = market_data.get("market_cap", 0)
        
        # Simple rule-based risk scoring
        return {
            "regulatory": 0.5,  # Default regulatory risk
            "market": min(1.0, 1 - (volume / 1e9)) if volume > 0 else 0.7,
            "technical": 0.7,  # Default technical risk
            "operational": 0.6,  # Default operational risk
            "fraud": 0.4 if price_change < 15 else 0.8  # Higher fraud risk with large price changes
        }
    
    def get_history(self) -> List[Dict]:
        """Return evaluation history"""
        return self.history