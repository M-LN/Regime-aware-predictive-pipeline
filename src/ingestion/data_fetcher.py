"""
Data ingestion module for fetching energy data from APIs
"""

import logging
import json
import os
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
        weather_endpoint: Optional[str] = "/weather",
        price_endpoint: str = "/prices",
        timestamp_field: str = "timestamp",
        start_param: str = "start",
        end_param: str = "end",
        field_map: Optional[Dict[str, str]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        energy_field_map: Optional[Dict[str, str]] = None,
        weather_field_map: Optional[Dict[str, str]] = None,
        price_field_map: Optional[Dict[str, str]] = None,
        energy_params: Optional[Dict[str, Any]] = None,
        weather_params: Optional[Dict[str, Any]] = None,
        price_params: Optional[Dict[str, Any]] = None,
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
        self.energy_field_map = energy_field_map
        self.weather_field_map = weather_field_map
        self.price_field_map = price_field_map
        self.energy_params = energy_params or {}
        self.weather_params = weather_params or {}
        self.price_params = price_params or {}
        self.logger = logging.getLogger(__name__)

    def fetch(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch energy, weather, and price data and merge on timestamp.
        """
        start_value = start_date.strftime("%Y-%m-%dT%H:%M")
        end_value = end_date.strftime("%Y-%m-%dT%H:%M")
        base_params = {
            self.start_param: start_value,
            self.end_param: end_value,
            **self.extra_params,
        }

        energy = self._fetch_endpoint(
            self.energy_endpoint,
            {**base_params, **self.energy_params},
            field_map=self.energy_field_map,
        )
        weather = self._fetch_endpoint(
            self.weather_endpoint,
            {**base_params, **self.weather_params},
            field_map=self.weather_field_map,
        )
        prices = self._fetch_endpoint(
            self.price_endpoint,
            {**base_params, **self.price_params},
            field_map=self.price_field_map,
        )

        if energy is None and prices is None and weather is None:
            raise RuntimeError("No data returned from EDS endpoints")

        merged = None
        if energy is not None and prices is not None:
            nearest_join = os.getenv("INGEST_NEAREST_JOIN", "true").lower() == "true"
            tolerance_raw = os.getenv("INGEST_NEAREST_TOLERANCE_MINUTES", "60")
            try:
                tolerance_minutes = int(tolerance_raw)
            except ValueError:
                tolerance_minutes = 60

            if nearest_join:
                energy_sorted = energy.sort_values("timestamp")
                price_sorted = prices.sort_values("timestamp")
                merged = pd.merge_asof(
                    energy_sorted,
                    price_sorted,
                    on="timestamp",
                    direction="nearest",
                    tolerance=pd.Timedelta(minutes=tolerance_minutes),
                )
            else:
                merged = pd.merge(energy, prices, on="timestamp", how="inner")
        elif energy is not None:
            merged = energy
        elif prices is not None:
            merged = prices

        if weather is not None:
            if merged is None:
                merged = weather
            else:
                merged = pd.merge(merged, weather, on="timestamp", how="left")

        merged = merged.sort_values("timestamp")
        merged = merged.reset_index(drop=True)

        ffill_columns = os.getenv("INGEST_FFILL_COLUMNS", "energy_production,price,temperature,wind_speed")
        ffill_limit_raw = os.getenv("INGEST_FFILL_LIMIT", "2")
        try:
            ffill_limit = int(ffill_limit_raw)
        except ValueError:
            ffill_limit = 2

        columns = [col.strip() for col in ffill_columns.split(",") if col.strip()]
        columns = [col for col in columns if col in merged.columns]
        if columns and ffill_limit > 0:
            merged[columns] = merged[columns].ffill(limit=ffill_limit)
        return merged

    def _fetch_endpoint(
        self,
        endpoint: Optional[str],
        params: Dict[str, Any],
        field_map: Optional[Dict[str, str]] = None,
    ) -> Optional[pd.DataFrame]:
        if not endpoint:
            return None

        url = f"{self.base_url}{endpoint}"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            request_params = {}
            for key, value in params.items():
                if isinstance(value, (dict, list)):
                    request_params[key] = json.dumps(value, ensure_ascii=True)
                else:
                    request_params[key] = value

            response = requests.get(url, params=request_params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as e:
            self.logger.warning("EDS request failed for %s: %s", url, e)
            return None

        if isinstance(payload, dict) and "records" in payload:
            records = payload["records"]
        elif isinstance(payload, dict) and "data" in payload:
            records = payload["data"]
        else:
            records = payload

        if not isinstance(records, list):
            self.logger.warning("Unexpected payload format from %s", url)
            return None

        df = pd.DataFrame(records)
        timestamp_field = self.timestamp_field
        if timestamp_field not in df.columns:
            for candidate in [
                "HourUTC",
                "HourDK",
                "Minutes5UTC",
                "Minutes5DK",
                "time",
                "TimeUTC",
                "TimeDK",
                "DateTime",
                "datetime",
            ]:
                if candidate in df.columns:
                    timestamp_field = candidate
                    self.logger.info("Using timestamp field '%s' for %s", candidate, url)
                    break

        if timestamp_field not in df.columns:
            self.logger.warning("Missing timestamp field '%s' in %s", self.timestamp_field, url)
            return None

        df = df.rename(columns={timestamp_field: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # Apply optional field mapping to standard schema
        mapping = field_map or self.field_map
        for target, source in mapping.items():
            if source in df.columns and target != source:
                df = df.rename(columns={source: target})

        # Heuristic mapping for common EDS column names
        if "price" not in df.columns:
            price_candidates = [
                "SpotPrice",
                "SpotPriceEUR",
                "SpotPriceDKK",
                "ElspotPrice",
                "Price",
            ]
            for candidate in price_candidates:
                if candidate in df.columns:
                    df = df.rename(columns={candidate: "price"})
                    break
        if "energy_production" not in df.columns:
            for col in df.columns:
                if "production" in col.lower():
                    df = df.rename(columns={col: "energy_production"})
                    break

        # Aggregate common EDS shapes to one row per timestamp
        if "price" in df.columns and "PriceArea" in df.columns:
            df = df.groupby("timestamp", as_index=False)["price"].mean()
        if "energy_production" in df.columns and "ProductionType" in df.columns:
            df = df.groupby("timestamp", as_index=False)["energy_production"].sum()

        # Keep standard columns if present
        keep_cols = ["timestamp"] + [c for c in mapping.keys() if c in df.columns]
        df = df[keep_cols].copy()

        return df


class OpenMeteoWeatherFetcher(DataFetcher):
    """
    Fetch weather data (temperature + wind speed) from Open-Meteo.
    """

    def __init__(
        self,
        base_url: str = "https://api.open-meteo.com/v1/forecast",
        latitude: float = 55.6761,
        longitude: float = 12.5683,
        timezone: str = "Europe/Copenhagen",
        hourly_fields: Optional[List[str]] = None,
        timeout: int = 30,
    ):
        self.base_url = base_url
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        self.hourly_fields = hourly_fields or ["temperature_2m", "wind_speed_10m"]
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    def fetch(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": ",".join(self.hourly_fields),
            "start_date": start_str,
            "end_date": end_str,
            "timezone": self.timezone,
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as e:
            self.logger.warning("Open-Meteo request failed: %s", e)
            return pd.DataFrame(columns=["timestamp", "temperature", "wind_speed"])

        hourly = payload.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        winds = hourly.get("wind_speed_10m", [])

        if not times:
            self.logger.warning("Open-Meteo returned no hourly data")
            return pd.DataFrame(columns=["timestamp", "temperature", "wind_speed"])

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(times, errors="coerce"),
            "temperature": temps,
            "wind_speed": winds,
        })
        return df


class CompositeDataFetcher(DataFetcher):
    """
    Combine multiple data sources and merge on timestamp.
    """

    def __init__(self, fetchers: List[DataFetcher]):
        self.fetchers = fetchers
        self.logger = logging.getLogger(__name__)

    def fetch(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        frames = []
        for fetcher in self.fetchers:
            frame = fetcher.fetch(start_date, end_date)
            if frame is not None and not frame.empty:
                frames.append(frame)

        if not frames:
            raise RuntimeError("No data returned from any data source")

        merged = frames[0]
        for frame in frames[1:]:
            merged = pd.merge(merged, frame, on="timestamp", how="outer")

        merged = merged.sort_values("timestamp")
        merged = merged.reset_index(drop=True)
        return merged


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
            
            safe_timestamp = date.strftime("%Y%m%dT%H%M%S")
            file_path = partition_path / f"data_{safe_timestamp}.parquet"
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
