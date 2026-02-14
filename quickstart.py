"""
Quick start script - Run this to get the system up and running
"""

import logging
from datetime import datetime, timedelta, UTC

from src.config import get_config
from src.ingestion.data_fetcher import MockEnergyDataFetcher, DataValidator, DataIngestionPipeline
from src.features.engineer import FeatureEngineer
from src.regime.detector import RegimeDetectionPipeline
from src.training import RegimeModelTrainer
from src.monitoring import initialize_mlflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main quickstart flow"""
    
    logger.info("Starting Regime-Aware Predictive Pipeline Setup...")
    logger.info("=" * 60)
    
    # Initialize MLflow for experiment tracking
    logger.info("\n[0/5] Initializing MLflow Experiment Tracking")
    logger.info("-" * 60)
    mlflow_tracker = initialize_mlflow()
    if mlflow_tracker.is_connected():
        logger.info("✓ MLflow tracking enabled (./mlruns)")
    else:
        logger.info("⚠ MLflow tracking disabled")
    
    # Step 1: Data Ingestion
    logger.info("\n[1/5] Data Ingestion")
    logger.info("-" * 60)
    
    fetcher = MockEnergyDataFetcher()
    validator = DataValidator()
    pipeline = DataIngestionPipeline(fetcher, validator)
    
    # Fetch 30 days of data
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=30)
    
    df_raw = pipeline.fetch_and_validate(start_date, end_date)
    
    if df_raw is None:
        logger.error("Failed to ingest data")
        return
    
    logger.info(f"✓ Ingested {len(df_raw)} records")
    logger.info(f"  Date range: {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}")
    logger.info(f"  Columns: {list(df_raw.columns)}")
    
    # Step 2: Feature Engineering
    logger.info("\n[2/4] Feature Engineering")
    logger.info("-" * 60)
    
    engineer = FeatureEngineer(rolling_windows=[1, 6, 24], volatility_windows=[6, 24])
    df_features = engineer.engineer_features(df_raw)
    
    logger.info(f"✓ Generated {len(df_features.columns)} features")
    logger.info(f"  Samples: {len(df_features)}")
    logger.info(f"  Features: {[col for col in df_features.columns if col not in df_raw.columns][:5]}...")
    
    # Step 3: Regime Detection
    logger.info("\n[3/5] Regime Detection (HMM Training)")
    logger.info("-" * 60)
    
    # Get feature columns for regime detection
    excluded_cols = ['timestamp', 'ingestion_timestamp', 'data_source', 'energy_production']
    feature_cols = [col for col in df_features.columns if col not in excluded_cols]
    
    regime_pipeline = RegimeDetectionPipeline(n_regimes=3)
    df_with_regime = regime_pipeline.fit_and_predict(df_features, feature_cols)
    
    logger.info(f"✓ Trained HMM model")
    logger.info(f"  Regime distribution:")
    for regime_id in range(3):
        count = (df_with_regime['regime'] == regime_id).sum()
        pct = (count / len(df_with_regime)) * 100
        logger.info(f"    Regime {regime_id}: {count} samples ({pct:.1f}%)")
    
    # Detect transitions
    transitions = regime_pipeline.detect_transitions(df_with_regime)
    logger.info(f"  Regime transitions: {len(transitions)}")
    
    # Save regime model
    regime_pipeline.save_model("data/models/regime_detector.pkl")
    logger.info(f"✓ Model saved to data/models/regime_detector.pkl")
    
    # Step 4: Per-Regime Model Training
    logger.info("\n[4/5] Per-Regime Model Training")
    logger.info("-" * 60)
    
    config = get_config()
    trainer = RegimeModelTrainer(config=config)
    results = trainer.train_all_regimes(df_with_regime, feature_cols, target_col="energy_production")
    
    if not results:
        logger.warning("No regime models were trained")
    else:
        for regime_id, result in results.items():
            logger.info(
                "✓ Regime %s model: %s | samples=%s | MAE=%.3f | RMSE=%.3f | %s",
                regime_id,
                result.model_name,
                result.n_samples,
                result.mae,
                result.rmse,
                result.model_path,
            )

    # Step 5: Inference Server
    logger.info("\n[5/5] Inference Server")
    logger.info("-" * 60)
    logger.info("To start the FastAPI inference server, run:")
    logger.info("")
    logger.info("  uvicorn src.inference.api:app --reload --port 8000")
    logger.info("")
    logger.info("Then test with:")
    logger.info("")
    logger.info("  curl -X POST http://localhost:8000/predict \\")
    logger.info("    -H 'Content-Type: application/json' \\")
    logger.info("    -d '{")
    logger.info('      "wind_speed": 8.5,')
    logger.info('      "energy_production": 400,')
    logger.info('      "temperature": 12,')
    logger.info('      "price": 150')
    logger.info("    }'")
    logger.info("")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SETUP COMPLETE ✓")
    logger.info("=" * 60)
    logger.info(f"Data samples: {len(df_raw)}")
    logger.info(f"Features generated: {len(df_features.columns)}")
    logger.info(f"Regime detection: Trained (3 regimes)")
    logger.info(f"Next: Start the API server (see instructions above)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
