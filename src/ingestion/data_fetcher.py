"""
Data ingestion module for fetching energy data from APIs
"""

import logging
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import pandas as pd
import requests
from pathlib import Path

logger = logging.getLogger(__name__)


class DataFetcher(ABC):
    """Abstract base class for data fetchers"""
    
    @abstractmethod
    def fetch(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data for given date range"""
        pass


class MockEnergyDataFetcher(DataFetcher):
    """
    Mock data fetcher for testing and development.
    Generates synthetic energy data with realistic patterns.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def fetch(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Generate synthetic energy data.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
        
        Returns:
            DataFrame with columns: timestamp, wind_speed, energy_production, 
                                   temperature, price
        """
        import numpy as np
        
        # Create hourly timestamps
        timestamps = pd.date_range(start=start_date, end=end_date, freq='h')
        n = len(timestamps)
        
        # Generate synthetic data with realistic patterns
        np.random.seed(42)
        
        # Wind speed (m/s) - varies with seasonal patterns
        wind_base = 6 + 2 * np.sin(np.arange(n) / 168)  # Weekly pattern
        wind_speed = wind_base + np.random.normal(0, 1, n)
        wind_speed = np.clip(wind_speed, 0, 20)
        
        # Energy production (MWh) - correlated with wind
        energy_production = 100 + 500 * (wind_speed / 20) + np.random.normal(0, 50, n)
        energy_production = np.clip(energy_production, 0, 1000)
        
        # Temperature (°C) - with daily and seasonal variation
        temp_base = 10 + 8 * np.sin(np.arange(n) / 24)
        temperature = temp_base + 3 * np.sin(np.arange(n) / (24 * 30)) + np.random.normal(0, 1, n)
        
        # Price (DKK/MWh) - inverse correlated with wind
        price_base = 250 - 100 * (wind_speed / 20)
        price = price_base + 50 * np.random.normal(1, 0.3, n)
        price = np.clip(price, 50, 500)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'wind_speed': wind_speed,
            'energy_production': energy_production,
            'temperature': temperature,
            'price': price
        })
        
        self.logger.info(f"Generated {len(df)} rows of synthetic energy data")
        
        return df


class EnergiDataServiceFetcher(DataFetcher):
    """
    Fetch data from Energi Data Service (EDS) or compatible APIs.

    This class assumes each endpoint returns JSON records with a timestamp field.
    Configure endpoints and field mappings via init params.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        energy_endpoint: str = "/energy",
        weather_endpoint: str = "/weather",
        price_endpoint: str = "/prices",
        timestamp_field: str = "timestamp",
        start_param: str = "start",
        end_param: str = "end",
        field_map: Optional[Dict[str, str]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.energy_endpoint = energy_endpoint
        self.weather_endpoint = weather_endpoint
        self.price_endpoint = price_endpoint
        self.timestamp_field = timestamp_field
        self.start_param = start_param
        self.end_param = end_param
        self.field_map = field_map or {
            "wind_speed": "wind_speed",
            "energy_production": "energy_production",
            "temperature": "temperature",
            "price": "price",
        }
        self.extra_params = extra_params or {}
        self.logger = logging.getLogger(__name__)

    def fetch(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch energy, weather, and price data and merge on timestamp.
        """
        params = {
            self.start_param: start_date.isoformat(),
            self.end_param: end_date.isoformat(),
            **self.extra_params,
        }

        energy = self._fetch_endpoint(self.energy_endpoint, params)
        weather = self._fetch_endpoint(self.weather_endpoint, params)
        prices = self._fetch_endpoint(self.price_endpoint, params)

        frames = []
        if energy is not None:
            frames.append(energy)
        if weather is not None:
            frames.append(weather)
        if prices is not None:
            frames.append(prices)

        if not frames:
            raise RuntimeError("No data returned from EDS endpoints")

        # Merge on timestamp with outer joins to preserve available data
        merged = frames[0]
        for frame in frames[1:]:
            merged = pd.merge(merged, frame, on="timestamp", how="outer")

        merged = merged.sort_values("timestamp")
        merged = merged.reset_index(drop=True)
        return merged

    def _fetch_endpoint(self, endpoint: str, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        url = f"{self.base_url}{endpoint}"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as e:
            self.logger.warning("EDS request failed for %s: %s", url, e)
            return None

        if isinstance(payload, dict) and "data" in payload:
            records = payload["data"]
        else:
            records = payload

        if not isinstance(records, list):
            self.logger.warning("Unexpected payload format from %s", url)
            return None

        df = pd.DataFrame(records)
        if self.timestamp_field not in df.columns:
            self.logger.warning("Missing timestamp field '%s' in %s", self.timestamp_field, url)
            return None

        df = df.rename(columns={self.timestamp_field: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # Apply optional field mapping to standard schema
        for target, source in self.field_map.items():
            if source in df.columns and target != source:
                df = df.rename(columns={source: target})

        # Keep standard columns if present
        keep_cols = ["timestamp"] + [c for c in self.field_map.keys() if c in df.columns]
        df = df[keep_cols].copy()

        return df


class DataValidator:
    """Validates data quality and schema"""
    
    def __init__(self, 
                 required_columns: List[str] = None,
                 null_threshold: float = 0.1):
        self.required_columns = required_columns or [
            'timestamp', 'wind_speed', 'energy_production', 'temperature', 'price'
        ]
        self.null_threshold = null_threshold
        self.logger = logging.getLogger(__name__)
    
    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validate DataFrame schema and quality.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            True if valid, False otherwise
        """
        # Check required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            self.logger.error(f"Missing columns: {missing_cols}")
            return False
        
        # Check for null values
        null_ratio = df.isnull().sum() / len(df)
        cols_with_high_nulls = null_ratio[null_ratio > self.null_threshold]
        if not cols_with_high_nulls.empty:
            self.logger.warning(f"High null ratio in columns: {cols_with_high_nulls.to_dict()}")
        
        # Check for duplicate timestamps
        if 'timestamp' in df.columns:
            duplicates = df['timestamp'].duplicated().sum()
            if duplicates > 0:
                self.logger.warning(f"Found {duplicates} duplicate timestamps")
        
        self.logger.info(f"Validation passed for {len(df)} records")
        return True


class DataIngestionPipeline:
    """Main ingestion pipeline orchestrator"""
    
    def __init__(self, 
                 fetcher: DataFetcher,
                 validator: DataValidator,
                 storage_path: str = "data/raw",
                 failed_path: str = "data/raw/failed"):
        self.fetcher = fetcher
        self.validator = validator
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.failed_path = Path(failed_path)
        self.failed_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def fetch_and_validate(self, 
                          start_date: datetime, 
                          end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Main pipeline: fetch → validate → return
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            Validated DataFrame or None if validation fails
        """
        try:
            # Fetch data
            self.logger.info(f"Fetching data from {start_date} to {end_date}")
            df = self.fetcher.fetch(start_date, end_date)
            
            # Validate
            if not self.validator.validate(df):
                self.logger.error("Data validation failed")
                self._save_failed_records(df, start_date, end_date, reason="validation_failed")
                return None
            
            # Add ingestion metadata
            df['ingestion_timestamp'] = datetime.utcnow()
            df['data_source'] = self.fetcher.__class__.__name__
            
            self.logger.info(f"Successfully ingested {len(df)} records")
            return df
        
        except Exception as e:
            self.logger.error(f"Ingestion pipeline failed: {e}")
            self._save_failed_records(None, start_date, end_date, reason=str(e))
            return None
    
    def save_parquet(self, df: pd.DataFrame, date: datetime) -> bool:
        """
        Save DataFrame to Parquet (partitioned by date).
        
        Args:
            df: DataFrame to save
            date: Date for partition
        
        Returns:
            True if successful
        """
        try:
            partition_path = self.storage_path / f"year={date.year}" / f"month={date.month}" / f"day={date.day}"
            partition_path.mkdir(parents=True, exist_ok=True)
            
            file_path = partition_path / f"data_{date.isoformat()}.parquet"
            df.to_parquet(file_path, engine='pyarrow', index=False)
            
            self.logger.info(f"Saved data to {file_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to save parquet: {e}")
            return False

    def _save_failed_records(
        self,
        df: Optional[pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        reason: str,
    ) -> None:
        """
        Persist failed ingestion payloads for inspection.
        """
        try:
            payload = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "reason": reason,
            }

            if df is not None:
                payload["records"] = df.to_dict(orient="records")

            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            file_path = self.failed_path / f"failed_{timestamp}.json"
            with file_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2, default=str)
            self.logger.info("Saved failed records to %s", file_path)
        except Exception as e:
            self.logger.warning("Failed to save failed records: %s", e)
