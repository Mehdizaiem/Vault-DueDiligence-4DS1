import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic market data and corresponding risk scores
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate features
    data = {
        'volume_24h': np.random.exponential(1e8, n_samples),
        'price_change_24h': np.random.normal(0, 5, n_samples),
        'market_cap': np.random.exponential(1e9, n_samples),
        'volatility': np.random.uniform(0.01, 0.5, n_samples),
        'liquidity_ratio': np.random.uniform(0.2, 2.0, n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add derived features
    df['price_volatility'] = np.abs(df['price_change_24h'])
    df['log_market_cap'] = np.log1p(df['market_cap'])
    
    # Generate target risk scores with some predefined relationships
    regulatory_risk = 0.3 + 0.4 * (1 - np.clip(df['market_cap'] / 5e9, 0, 1)) + np.random.normal(0, 0.1, n_samples)
    market_risk = 0.2 + 0.6 * df['volatility'] + 0.2 * (1 - np.clip(df['volume_24h'] / 5e8, 0, 1)) + np.random.normal(0, 0.1, n_samples)
    technical_risk = 0.4 + 0.3 * df['volatility'] + 0.3 * (1 - df['liquidity_ratio'] / 2) + np.random.normal(0, 0.1, n_samples)
    operational_risk = 0.3 + 0.2 * df['price_volatility'] / 10 + 0.5 * (1 - np.clip(df['market_cap'] / 2e9, 0, 1)) + np.random.normal(0, 0.1, n_samples)
    fraud_risk = 0.2 + 0.5 * np.clip(df['price_volatility'] / 15, 0, 1) + 0.3 * (1 - np.clip(df['volume_24h'] / 3e8, 0, 1)) + np.random.normal(0, 0.1, n_samples)
    
    # Clip all risks to [0,1] range
    df['regulatory_risk'] = np.clip(regulatory_risk, 0, 1)
    df['market_risk'] = np.clip(market_risk, 0, 1)
    df['technical_risk'] = np.clip(technical_risk, 0, 1)
    df['operational_risk'] = np.clip(operational_risk, 0, 1)
    df['fraud_risk'] = np.clip(fraud_risk, 0, 1)
    
    return df

def train_model(data, output_dir=None):
    """
    Train a model to predict risk scores from market data
    """
    # Define features and targets
    feature_cols = ['volume_24h', 'price_change_24h', 'market_cap', 'volatility', 
                    'liquidity_ratio', 'price_volatility', 'log_market_cap']
    target_cols = ['regulatory_risk', 'market_risk', 'technical_risk', 
                   'operational_risk', 'fraud_risk']
    
    X = data[feature_cols]
    y = data[target_cols]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    logger.info("Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Model performance - MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # Feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    logger.info("Feature importance:")
    for i, row in feature_importance.iterrows():
        logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    # Save model and scaler
    if not output_dir:
        output_dir = Path(__file__).parent / "models"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    model_path = output_dir / "risk_model.pkl"
    scaler_path = output_dir / "scaler.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Scaler saved to {scaler_path}")
    
    return model, scaler, (model_path, scaler_path)

if __name__ == "__main__":
    # Generate synthetic training data
    logger.info("Generating synthetic training data...")
    df = generate_synthetic_data(n_samples=2000)
    
    # Train and save model
    model_dir = Path(__file__).parent / "models"
    model, scaler, (model_path, _) = train_model(df, model_dir)
    
    logger.info("Done! You can now use the model with MLRiskEvaluator.")