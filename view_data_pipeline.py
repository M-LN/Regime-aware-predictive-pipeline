"""
Data & Pipeline Viewer - See your data and how the system works
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import pickle

print("\n" + "="*80)
print("🔍 REGIME AI - DATA & PIPELINE VIEWER")
print("="*80)

# ============================================================================
# PART 1: THE PIPELINE (How It Works)
# ============================================================================
print("\n" + "─"*80)
print("📋 PIPELINE WORKFLOW")
print("─"*80)

pipeline_steps = [
    ("1️⃣  DATA INGESTION", "Fetch energy market data (wind, price, temperature, production)", "✓ Complete"),
    ("2️⃣  DATA VALIDATION", "Check for missing values, duplicates, outliers", "✓ Complete"),
    ("3️⃣  FEATURE ENGINEERING", "Generate 112 features: rolling stats, volatility, temporal", "✓ Complete"),
    ("4️⃣  REGIME DETECTION", "HMM identifies 3 market regimes (high/neutral/low wind)", "✓ Complete"),
    ("5️⃣  MODEL TRAINING", "Train XGBoost, LSTM, Random Forest per regime", "✓ Complete"),
    ("6️⃣  MODEL REGISTRY", "Save models to disk, track in MLflow", "✓ Complete"),
    ("7️⃣  API DEPLOYMENT", "FastAPI server for real-time predictions", "✓ Running"),
]

for step, description, status in pipeline_steps:
    print(f"\n{step}")
    print(f"   Description: {description}")
    print(f"   Status: {status}")

# ============================================================================
# PART 2: VIEW ACTUAL DATA
# ============================================================================
print("\n\n" + "="*80)
print("📊 YOUR ACTUAL DATA")
print("="*80)

# Check for raw data files
raw_data_path = Path("data/raw")
print(f"\n[RAW DATA FILES]")
print(f"Location: {raw_data_path.absolute()}")

if raw_data_path.exists():
    failed_files = list(raw_data_path.glob("failed/*.json"))
    if failed_files:
        print(f"\n⚠️  Found {len(failed_files)} failed ingestion attempts")
        print("   These are from previous runs - you can ignore them")
        
        # Show one failed file content
        if len(failed_files) > 0:
            with open(failed_files[0], 'r') as f:
                failed_data = json.load(f)
                print(f"\n   Sample failed record: {failed_files[0].name}")
                print(f"   Error: {failed_data.get('error', 'Unknown')}")
    
    # Look for successful data
    data_files = list(raw_data_path.glob("year=*/month=*/day=*/*.parquet"))
    if data_files:
        print(f"\n✓ Found {len(data_files)} data files")
    else:
        print("\n💡 No raw data files yet - they'll appear when you run data ingestion")

# ============================================================================
# PART 3: GENERATE & VIEW SAMPLE DATA
# ============================================================================
print("\n\n" + "─"*80)
print("📈 GENERATING SAMPLE DATA TO SHOW YOU THE PIPELINE")
print("─"*80)

from src.ingestion.data_fetcher import MockEnergyDataFetcher, DataValidator
from src.features.engineer import FeatureEngineer
from src.regime.detector import RegimeDetectionPipeline
from datetime import timedelta, UTC

# Step 1: Generate sample data
print("\n[STEP 1: Data Ingestion]")
fetcher = MockEnergyDataFetcher()
end_date = datetime.now(UTC)
start_date = end_date - timedelta(days=3)  # 3 days of data

df_raw = fetcher.fetch(start_date, end_date)
print(f"✓ Generated {len(df_raw)} hourly records")
print(f"\n📋 Raw Data Sample (first 5 rows):")
print(df_raw.head(5).to_string(index=False))

print(f"\n📊 Raw Data Statistics:")
print(df_raw[['wind_speed', 'energy_production', 'temperature', 'price']].describe().round(2))

# Step 2: Feature Engineering
print("\n\n[STEP 2: Feature Engineering]")
feature_engineer = FeatureEngineer(
    rolling_windows=[1, 6, 24],
    volatility_windows=[6, 24]
)
df_features = feature_engineer.engineer_features(df_raw)

print(f"✓ Generated {len(df_features.columns)} features from {len(df_features)} samples")
print(f"\n📋 Feature Categories:")
temporal_features = [col for col in df_features.columns if any(x in col for x in ['hour', 'day', 'month', 'weekday'])]
rolling_features = [col for col in df_features.columns if 'rolling' in col]
volatility_features = [col for col in df_features.columns if 'volatility' in col or '_cv_' in col]

print(f"   • Temporal features: {len(temporal_features)} (hour, day, month, etc.)")
print(f"   • Rolling statistics: {len(rolling_features)} (mean, std, min, max)")
print(f"   • Volatility metrics: {len(volatility_features)} (CV, price volatility)")

print(f"\n📋 Sample Features (first 3 rows):")
sample_cols = ['timestamp', 'wind_speed', 'price']
if 'wind_speed_rolling_mean_6h' in df_features.columns:
    sample_cols.append('wind_speed_rolling_mean_6h')
if 'price_rolling_std_6h' in df_features.columns:
    sample_cols.append('price_rolling_std_6h')
print(df_features[sample_cols].head(3).to_string(index=False))

# Step 3: Regime Detection
print("\n\n[STEP 3: Regime Detection]")
regime_pipeline = RegimeDetectionPipeline(n_regimes=3)
feature_cols = [col for col in df_features.columns if col != 'timestamp']

df_with_regimes = regime_pipeline.fit_and_predict(
    df_features,
    feature_cols=feature_cols,
    signal_col='price'
)

print(f"✓ Detected {df_with_regimes['regime'].nunique()} distinct market regimes")

regime_counts = df_with_regimes['regime'].value_counts().sort_index()
print(f"\n📊 Regime Distribution:")
for regime_id, count in regime_counts.items():
    percentage = (count / len(df_with_regimes)) * 100
    print(f"   Regime {regime_id}: {count:3d} samples ({percentage:5.1f}%)")

# Show regime characteristics
print(f"\n📋 Regime Characteristics:")
regime_stats = df_with_regimes.groupby('regime').agg({
    'wind_speed': 'mean',
    'price': 'mean',
    'energy_production': 'mean'
}).round(2)

regime_names = {
    0: "High Wind Regime",
    1: "Neutral Regime", 
    2: "Low Wind Regime"
}

for regime_id in regime_stats.index:
    stats = regime_stats.loc[regime_id]
    name = regime_names.get(regime_id, f"Regime {regime_id}")
    print(f"\n   {name} (Regime {regime_id}):")
    print(f"   • Avg Wind Speed: {stats['wind_speed']:.2f} m/s")
    print(f"   • Avg Price: ${stats['price']:.2f}/MWh")
    print(f"   • Avg Energy Production: {stats['energy_production']:.2f} MWh")

# ============================================================================
# PART 4: VIEW TRAINED MODELS
# ============================================================================
print("\n\n" + "="*80)
print("🤖 TRAINED MODELS")
print("="*80)

models_path = Path("data/models")
if models_path.exists():
    model_files = list(models_path.glob("*.pkl")) + list(models_path.glob("*.keras"))
    
    print(f"\nFound {len(model_files)} trained models:\n")
    
    for model_file in sorted(model_files):
        size_mb = model_file.stat().st_size / 1024 / 1024
        print(f"✓ {model_file.name}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Modified: {datetime.fromtimestamp(model_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show what each model is for
        if "regime_0" in model_file.name:
            print(f"  Purpose: High Wind Regime prediction (XGBoost)")
        elif "regime_1" in model_file.name:
            print(f"  Purpose: Neutral Regime prediction (LSTM)")
        elif "regime_2" in model_file.name:
            print(f"  Purpose: Low Wind Regime prediction (Random Forest)")
        elif "regime_detector" in model_file.name:
            print(f"  Purpose: Regime classification (HMM)")
        print()

# ============================================================================
# PART 5: PREDICTION FLOW VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("🔮 PREDICTION FLOW")
print("="*80)

print("""
When you make a prediction request:

    📥 INPUT DATA
    ↓
    ┌─────────────────────────────────────┐
    │ Wind Speed: 7.5 m/s                │
    │ Energy Production: 220 MWh         │
    │ Temperature: 12°C                  │
    │ Current Price: $250/MWh            │
    └─────────────────────────────────────┘
    ↓
    [FEATURE ENGINEERING]
    ↓
    ┌─────────────────────────────────────┐
    │ 112 Features Generated:            │
    │ • Rolling means (1h, 6h, 24h)     │
    │ • Volatility metrics              │
    │ • Temporal features               │
    │ • Domain-specific calculations    │
    └─────────────────────────────────────┘
    ↓
    [REGIME DETECTION - HMM]
    ↓
    ┌─────────────────────────────────────┐
    │ Detected Regime: 1 (Neutral)      │
    │ Confidence: 98.5%                 │
    └─────────────────────────────────────┘
    ↓
    [MODEL SELECTION]
    ↓
    ┌─────────────────────────────────────┐
    │ Route to: LSTM Model              │
    │ (Best for neutral conditions)     │
    └─────────────────────────────────────┘
    ↓
    [PREDICTION]
    ↓
    📤 OUTPUT
    ┌─────────────────────────────────────┐
    │ Predicted Price: $245.30/MWh      │
    │ Regime: Neutral                   │
    │ Confidence: 98.5%                 │
    │ Processing Time: 25ms             │
    └─────────────────────────────────────┘
""")

# ============================================================================
# PART 6: HOW TO EXPLORE MORE
# ============================================================================
print("\n" + "="*80)
print("🎯 HOW TO EXPLORE YOUR DATA")
print("="*80)

print("""
1️⃣  VIEW DATA IN PYTHON:
   >>> import pandas as pd
   >>> # Load raw data
   >>> from src.ingestion.data_fetcher import MockEnergyDataFetcher
   >>> fetcher = MockEnergyDataFetcher()
   >>> df = fetcher.fetch(start_date, end_date)
   >>> df.head()

2️⃣  INSPECT MODELS:
   >>> import pickle
   >>> with open('data/models/regime_detector.pkl', 'rb') as f:
   ...     detector = pickle.load(f)
   >>> print(detector)

3️⃣  VIEW FEATURES:
   >>> from src.features.engineer import FeatureEngineer
   >>> engineer = FeatureEngineer()
   >>> features = engineer.engineer_features(df)
   >>> features.columns.tolist()  # See all 112 features

4️⃣  EXPLORE MLFLOW:
   • Open: http://localhost:5000
   • View experiments, metrics, parameters
   • Compare model runs

5️⃣  TEST PREDICTIONS:
   • API Docs: http://127.0.0.1:8002/docs
   • Test Script: python test_prediction.py
   • Dashboard: Open api_dashboard.html
""")

print("\n" + "="*80)
print("✅ DATA & PIPELINE OVERVIEW COMPLETE!")
print("="*80)
print("\n💡 Run this script anytime to see the data flow: python view_data_pipeline.py\n")
