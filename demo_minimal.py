"""
Minimal Demo - Runs the core system without heavy ML libraries
"""

import logging
from datetime import datetime, timedelta, UTC
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo():
    """Minimal demo of the system"""
    
    logger.info("Starting Regime-Aware Predictive Pipeline DEMO...")
    logger.info("=" * 60)
    
    # Step 1: Import and test core modules
    logger.info("\n[1/3] Testing Core Imports")
    logger.info("-" * 60)
    
    try:
        from src.config import Config, load_config
        logger.info("✓ Configuration module loaded")
    except Exception as e:
        logger.error(f"✗ Failed to load config: {e}")
        return
    
    try:
        from src.ingestion.data_fetcher import MockEnergyDataFetcher, DataValidator
        logger.info("✓ Ingestion module loaded")
    except Exception as e:
        logger.error(f"✗ Failed to load ingestion: {e}")
        return
    
    try:
        from src.features.engineer import FeatureEngineer
        logger.info("✓ Feature engineering module loaded")
    except Exception as e:
        logger.error(f"✗ Failed to load features: {e}")
        return
    
    # Step 2: Data Ingestion
    logger.info("\n[2/3] Data Ingestion & Feature Engineering")
    logger.info("-" * 60)
    
    try:
        fetcher = MockEnergyDataFetcher()
        validator = DataValidator()
        
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=7)  # Use 7 days instead of 30 for speed
        
        df_raw = fetcher.fetch(start_date, end_date)
        validated = validator.validate(df_raw)
        
        logger.info(f"✓ Fetched {len(df_raw)} records")
        logger.info(f"  Columns: {list(df_raw.columns)}")
        logger.info(f"  Date range: {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}")
        
        # Feature Engineering
        engineer = FeatureEngineer(rolling_windows=[1, 6], volatility_windows=[6])
        df_features = engineer.engineer_features(df_raw)
        
        logger.info(f"✓ Generated {len(df_features.columns)} features from {len(df_features)} samples")
        
    except Exception as e:
        logger.error(f"✗ Data pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: System Summary
    logger.info("\n[3/3] System Status")
    logger.info("-" * 60)
    
    try:
        config = load_config()
        logger.info(f"✓ Configuration loaded ({config.environment})")
        logger.info(f"  • API: {config.inference.host}:{config.inference.port}")
        logger.info(f"  • Regimes: {config.regime.n_regimes}")
        logger.info(f"  • Models: {config.model.regime_a_model}, {config.model.regime_b_model}, {config.model.regime_c_model}")
    except Exception as e:
        logger.warning(f"⚠ Could not load config: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("DEMO COMPLETE ✓")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next Steps:")
    logger.info("  1. Install heavy dependencies: pip install -r requirements.txt")
    logger.info("  2. Run full quickstart: python quickstart.py")
    logger.info("  3. Start API server: uvicorn src.inference.api:app --reload")
    logger.info("  4. Test prediction: curl -X POST http://localhost:8000/predict ...")
    logger.info("")


if __name__ == "__main__":
    demo()
