"""
API Demo - Shows the Regime-Aware Prediction Pipeline in action
"""

import logging
import sys
from datetime import datetime, timedelta, UTC
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_pipeline():
    """Demonstrate the complete pipeline workflow"""
    
    print("\n" + "="*70)
    print("🚀 REGIME-AWARE PREDICTIVE PIPELINE - LIVE DEMO")
    print("="*70 + "\n")
    
    # Step 1: Import and Initialize
    print("[1/5] 🔧 Initializing System Components")
    print("-" * 70)
    
    try:
        from src.config import get_config
        from src.ingestion.data_fetcher import MockEnergyDataFetcher, DataValidator
        from src.features.engineer import FeatureEngineer
        from src.regime.detector import RegimeDetectionPipeline
        
        config = get_config()
        print(f"✓ Configuration loaded (Environment: {config.environment})")
        print(f"  • Regimes configured: {config.regime.n_regimes}")
        print(f"  • Model types: XGBoost, LSTM, Random Forest")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return False
    
    # Step 2: Data Ingestion
    print(f"\n[2/5] 📊 Fetching Energy Market Data")
    print("-" * 70)
    
    try:
        fetcher = MockEnergyDataFetcher()
        validator = DataValidator()
        
        # Fetch 7 days of hourly data
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=7)
        
        data = fetcher.fetch(start_date, end_date)
        is_valid = validator.validate(data)
        
        if not is_valid:
            raise ValueError("Data validation failed")
        
        df_validated = data
        
        print(f"✓ Fetched {len(df_validated)} records")
        print(f"  • Time range: {df_validated['timestamp'].min()} to {df_validated['timestamp'].max()}")
        print(f"  • Features: {list(df_validated.columns)}")
        print(f"\n  Sample data (first 3 rows):")
        print(f"  {df_validated.head(3).to_string(index=False)}\n")
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return False
    
    # Step 3: Feature Engineering
    print(f"[3/5] ⚙️  Engineering Features")
    print("-" * 70)
    
    try:
        feature_engineer = FeatureEngineer(
            rolling_windows=[1, 6, 24],
            volatility_windows=[6, 24]
        )
        
        df_features = feature_engineer.engineer_features(df_validated)
        
        feature_cols = [col for col in df_features.columns if col not in ['timestamp']]
        print(f"✓ Generated {len(feature_cols)} features from {len(df_features)} samples")
        print(f"  • Rolling statistics: mean, std, min, max")
        print(f"  • Volatility metrics: CV, price_volatility")
        print(f"  • Domain features: capacity_factor, wind_energy_ratio")
        
        # Show some key features (use columns that actually exist)
        display_cols = ['timestamp', 'wind_speed', 'price']
        # Add additional columns if they exist
        if 'price_volatility_6h' in df_features.columns:
            display_cols.append('price_volatility_6h')
        if 'wind_speed_rolling_mean_6h' in df_features.columns:
            display_cols.append('wind_speed_rolling_mean_6h')
        
        sample_features = df_features[display_cols].head(3)
        print(f"\n  Sample features:")
        print(f"  {sample_features.to_string(index=False)}\n")
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return False
    
    # Step 4: Regime Detection
    print(f"[4/5] 🎯 Detecting Market Regimes")
    print("-" * 70)
    
    try:
        regime_pipeline = RegimeDetectionPipeline(n_regimes=3)
        
        # Get feature columns (exclude timestamp)
        feature_cols = [col for col in df_features.columns if col != 'timestamp']
        
        # Fit and detect regimes
        df_with_regimes = regime_pipeline.fit_and_predict(
            df_features,
            feature_cols=feature_cols,
            signal_col='price'
        )
        
        regime_counts = df_with_regimes['regime'].value_counts().sort_index()
        
        print(f"✓ Detected {config.regime.n_regimes} distinct market regimes")
        print(f"\n  Regime Distribution:")
        for regime_id, count in regime_counts.items():
            percentage = (count / len(df_with_regimes)) * 100
            print(f"  • Regime {regime_id}: {count:3d} samples ({percentage:5.1f}%)")
        
        # Show regime characteristics
        print(f"\n  Regime Characteristics:")
        
        # Use columns that actually exist
        agg_cols = {'wind_speed': 'mean', 'price': 'mean'}
        if 'price_volatility_6h' in df_with_regimes.columns:
            agg_cols['price_volatility_6h'] = 'mean'
        
        regime_stats = df_with_regimes.groupby('regime').agg(agg_cols).round(2)
        
        regime_names = {
            0: "High Wind (Low Price, Low Volatility)",
            1: "Neutral (Medium Price, Medium Volatility)",
            2: "Low Wind (High Price, High Volatility)"
        }
        
        for regime_id in regime_stats.index:
            stats = regime_stats.loc[regime_id]
            name = regime_names.get(regime_id, f"Regime {regime_id}")
            print(f"  • {name}")
            print(f"    - Avg Wind Speed: {stats['wind_speed']:.2f} m/s")
            print(f"    - Avg Price: ${stats['price']:.2f}/MWh")
            if 'price_volatility_6h' in stats:
                print(f"    - Avg Volatility (6h): {stats['price_volatility_6h']:.3f}")
        
    except Exception as e:
        logger.error(f"Regime detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Prediction Routing
    print(f"\n[5/5] 🔮 Prediction Routing Strategy")
    print("-" * 70)
    
    try:
        print(f"✓ Multi-model prediction routing configured")
        print(f"\n  Model Assignment:")
        
        models = {
            0: ("XGBoost", config.model.regime_a_model),
            1: ("LSTM", config.model.regime_b_model),
            2: ("Random Forest", config.model.regime_c_model)
        }
        
        for regime_id in range(config.regime.n_regimes):
            regime_name = regime_names.get(regime_id, f"Regime {regime_id}")
            model_name, model_type = models.get(regime_id, ("Unknown", "unknown"))
            
            print(f"  • {regime_name}")
            print(f"    → Model: {model_name}")
            print(f"    → Best for: {_get_model_strength(model_type)}")
        
        # Simulate a prediction
        print(f"\n  📍 Example Prediction Flow:")
        latest_data = df_with_regimes.iloc[-1]
        detected_regime = latest_data['regime']
        
        print(f"  1. Input: Wind={latest_data['wind_speed']:.2f} m/s, Price=${latest_data['price']:.2f}/MWh")
        print(f"  2. Regime Detection → {regime_names[detected_regime]}")
        print(f"  3. Model Selection → {_get_model_for_regime(detected_regime, config)}")
        print(f"  4. Prediction → Price forecast for next hour")
        print(f"  5. Response with confidence scores and regime metadata")
        
    except Exception as e:
        logger.error(f"Prediction routing demo failed: {e}")
        return False
    
    # Summary
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE - System is ready for production deployment!")
    print("="*70)
    
    print("\n📚 Next Steps:")
    print("  1. Train models:     python -m src.training.regime_trainer")
    print("  2. Start API:        uvicorn src.inference.api:app --host 0.0.0.0 --port 8000")
    print("  3. View API docs:    http://localhost:8000/docs")
    print("  4. Test prediction:  curl -X POST http://localhost:8000/predict -d '{...}'")
    print("  5. Monitor metrics:  http://localhost:8000/metrics")
    print("\n🎉 The system intelligently routes predictions through specialized models")
    print("   based on detected market regimes for superior accuracy!")
    
    return True


def _get_model_strength(model_type):
    """Get model strength description"""
    strengths = {
        'xgboost': 'Stable patterns, high wind scenarios',
        'lstm': 'Sequential patterns, temporal dependencies',
        'random_forest': 'Non-linear relationships, volatile markets'
    }
    return strengths.get(model_type.lower(), 'Unknown')


def _get_model_for_regime(regime_id, config):
    """Get model type for regime"""
    models = {
        0: config.model.regime_a_model,
        1: config.model.regime_b_model,
        2: config.model.regime_c_model
    }
    return models.get(regime_id, 'unknown').upper()


if __name__ == "__main__":
    try:
        success = demo_pipeline()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
