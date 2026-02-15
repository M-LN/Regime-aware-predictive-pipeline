# Data Exploration Notebook Commands

# To explore your data interactively, run these commands in Python:

# 1. Load raw data
from src.ingestion.data_fetcher import MockEnergyDataFetcher
from datetime import datetime, timedelta, UTC
import pandas as pd

fetcher = MockEnergyDataFetcher()
end_date = datetime.now(UTC)
start_date = end_date - timedelta(days=7)
df_raw = fetcher.fetch(start_date, end_date)

# 2. View data
print("\n=== RAW DATA ===")
print(df_raw.head(10))
print("\n=== DATA STATISTICS ===")
print(df_raw.describe())

# 3. Load engineered features
from src.features.engineer import FeatureEngineer

engineer = FeatureEngineer()
df_features = engineer.engineer_features(df_raw)

print("\n=== ENGINEERED FEATURES ===")
print(f"Total features: {len(df_features.columns)}")
print("Feature names:", df_features.columns.tolist())

# 4. View regime detection
from src.regime.detector import RegimeDetectionPipeline

detector = RegimeDetectionPipeline(n_regimes=3)
feature_cols = [col for col in df_features.columns if col != 'timestamp']
df_regimes = detector.fit_and_predict(df_features, feature_cols, signal_col='price')

print("\n=== REGIME DISTRIBUTION ===")
print(df_regimes['regime'].value_counts())

print("\n=== REGIME CHARACTERISTICS ===")
print(df_regimes.groupby('regime')[['wind_speed', 'price', 'energy_production']].mean())

# 5. Load trained models
import pickle

with open('data/models/regime_detector.pkl', 'rb') as f:
    regime_model = pickle.load(f)
    print("\n=== REGIME DETECTOR MODEL ===")
    print(regime_model)

with open('data/models/regime_0_xgboost.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
    print("\n=== XGBOOST MODEL ===")
    print(xgb_model)

print("\n✅ All data loaded successfully!")
print("Run each section separately to explore the data interactively")
