"""
Unit tests for ingestion module
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.ingestion.data_fetcher import (
    MockEnergyDataFetcher,
    DataValidator,
    DataIngestionPipeline
)


@pytest.fixture
def data_fetcher():
    return MockEnergyDataFetcher()


@pytest.fixture
def data_validator():
    return DataValidator()


@pytest.fixture
def ingestion_pipeline(data_fetcher, data_validator):
    return DataIngestionPipeline(data_fetcher, data_validator)


class TestMockDataFetcher:
    
    def test_fetch_returns_dataframe(self, data_fetcher):
        """Test that fetcher returns DataFrame"""
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 2)
        
        df = data_fetcher.fetch(start, end)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 25  # 25 hours
    
    def test_fetch_has_required_columns(self, data_fetcher):
        """Test that fetched data has required columns"""
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 2)
        
        df = data_fetcher.fetch(start, end)
        required = ['timestamp', 'wind_speed', 'energy_production', 'temperature', 'price']
        
        for col in required:
            assert col in df.columns
    
    def test_fetch_data_in_valid_ranges(self, data_fetcher):
        """Test that fetched data is in realistic ranges"""
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 7)
        
        df = data_fetcher.fetch(start, end)
        
        assert (df['wind_speed'] >= 0).all()
        assert (df['wind_speed'] <= 20).all()
        assert (df['energy_production'] >= 0).all()
        assert (df['energy_production'] <= 1000).all()
        assert (df['temperature'] >= -40).all()
        assert (df['temperature'] <= 50).all()
        assert (df['price'] >= 0).all()


class TestDataValidator:
    
    def test_validate_valid_data(self, data_validator):
        """Test validation passes for good data"""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2026-01-01', periods=10, freq='h'),
            'wind_speed': np.random.rand(10) * 20,
            'energy_production': np.random.rand(10) * 1000,
            'temperature': np.random.rand(10) * 50,
            'price': np.random.rand(10) * 500
        })
        
        assert data_validator.validate(df) == True
    
    def test_validate_missing_columns(self, data_validator):
        """Test validation fails for missing columns"""
        df = pd.DataFrame({'timestamp': [1, 2, 3]})
        
        assert data_validator.validate(df) == False
    
    def test_validate_high_null_ratio(self, data_validator):
        """Test validation handles high null ratios"""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2026-01-01', periods=10, freq='h'),
            'wind_speed': [np.nan] * 10,
            'energy_production': np.random.rand(10) * 1000,
            'temperature': np.random.rand(10) * 50,
            'price': np.random.rand(10) * 500
        })
        
        # Should still return True but log warning
        assert data_validator.validate(df) == True


class TestIngestionPipeline:
    
    def test_pipeline_integration(self, ingestion_pipeline):
        """Test full ingestion pipeline"""
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 2)
        
        df = ingestion_pipeline.fetch_and_validate(start, end)
        
        assert df is not None
        assert len(df) > 0
        assert 'ingestion_timestamp' in df.columns
        assert 'data_source' in df.columns
