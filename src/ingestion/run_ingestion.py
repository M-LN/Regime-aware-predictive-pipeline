"""
Run data ingestion for a given time range.
"""

import argparse
from datetime import datetime, timedelta, timezone
import os
import json
import logging
import pandas as pd
import requests

from src.config import get_config
from src.ingestion.data_fetcher import (
    DataIngestionPipeline,
    DataValidator,
    MockEnergyDataFetcher,
    EnergiDataServiceFetcher,
    OpenMeteoWeatherFetcher,
    CompositeDataFetcher,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ingestion for a time range")
    parser.add_argument("--start", help="ISO timestamp start (default: now - lookback)")
    parser.add_argument("--end", help="ISO timestamp end (default: now)")
    parser.add_argument("--lookback-hours", type=int, default=None)
    return parser.parse_args()


def _build_fetcher(config) -> object:
    base_url = os.getenv("EDS_BASE_URL")
    if not base_url:
        return MockEnergyDataFetcher()

    field_map = None
    extra_params = None
    energy_field_map = None
    price_field_map = None
    energy_params = None
    price_params = None

    field_map_raw = os.getenv("EDS_FIELD_MAP")
    if field_map_raw:
        try:
            field_map = json.loads(field_map_raw)
        except Exception as e:
            logger.warning("Invalid EDS_FIELD_MAP JSON: %s", e)

    energy_field_map_raw = os.getenv("EDS_ENERGY_FIELD_MAP")
    if energy_field_map_raw:
        try:
            energy_field_map = json.loads(energy_field_map_raw)
        except Exception as e:
            logger.warning("Invalid EDS_ENERGY_FIELD_MAP JSON: %s", e)

    price_field_map_raw = os.getenv("EDS_PRICE_FIELD_MAP")
    if price_field_map_raw:
        try:
            price_field_map = json.loads(price_field_map_raw)
        except Exception as e:
            logger.warning("Invalid EDS_PRICE_FIELD_MAP JSON: %s", e)

    extra_params_raw = os.getenv("EDS_EXTRA_PARAMS")
    if extra_params_raw:
        try:
            extra_params = json.loads(extra_params_raw)
        except Exception as e:
            logger.warning("Invalid EDS_EXTRA_PARAMS JSON: %s", e)

    energy_params_raw = os.getenv("EDS_ENERGY_EXTRA_PARAMS")
    if energy_params_raw:
        try:
            energy_params = json.loads(energy_params_raw)
        except Exception as e:
            logger.warning("Invalid EDS_ENERGY_EXTRA_PARAMS JSON: %s", e)

    price_params_raw = os.getenv("EDS_PRICE_EXTRA_PARAMS")
    if price_params_raw:
        try:
            price_params = json.loads(price_params_raw)
        except Exception as e:
            logger.warning("Invalid EDS_PRICE_EXTRA_PARAMS JSON: %s", e)

    eds_fetcher = EnergiDataServiceFetcher(
        base_url=base_url,
        api_key=os.getenv("EDS_API_KEY"),
        timeout=int(os.getenv("EDS_TIMEOUT", config.data.api_timeout)),
        energy_endpoint=os.getenv("EDS_ENERGY_ENDPOINT", "/energy"),
        weather_endpoint=os.getenv("EDS_WEATHER_ENDPOINT", ""),
        price_endpoint=os.getenv("EDS_PRICE_ENDPOINT", "/prices"),
        timestamp_field=os.getenv("EDS_TIMESTAMP_FIELD", "timestamp"),
        start_param=os.getenv("EDS_START_PARAM", "start"),
        end_param=os.getenv("EDS_END_PARAM", "end"),
        field_map=field_map,
        extra_params=extra_params,
        energy_field_map=energy_field_map,
        price_field_map=price_field_map,
        energy_params=energy_params,
        price_params=price_params,
    )

    weather_base_url = os.getenv("WEATHER_BASE_URL")
    if not weather_base_url:
        return eds_fetcher

    weather_fetcher = OpenMeteoWeatherFetcher(
        base_url=weather_base_url,
        latitude=float(os.getenv("WEATHER_LATITUDE", "55.6761")),
        longitude=float(os.getenv("WEATHER_LONGITUDE", "12.5683")),
        timezone=os.getenv("WEATHER_TIMEZONE", "Europe/Copenhagen"),
        hourly_fields=os.getenv(
            "WEATHER_HOURLY_FIELDS", "temperature_2m,wind_speed_10m"
        ).split(","),
        timeout=int(os.getenv("WEATHER_TIMEOUT", "30")),
    )

    return CompositeDataFetcher([eds_fetcher, weather_fetcher])


def _get_latest_eds_timestamp() -> datetime | None:
    base_url = os.getenv("EDS_BASE_URL")
    price_endpoint = os.getenv("EDS_PRICE_ENDPOINT")
    if not base_url or not price_endpoint:
        return None

    timestamp_field = os.getenv("EDS_TIMESTAMP_FIELD", "timestamp")
    price_params = {}
    price_params_raw = os.getenv("EDS_PRICE_EXTRA_PARAMS")
    if price_params_raw:
        try:
            price_params = json.loads(price_params_raw)
        except Exception as e:
            logger.warning("Invalid EDS_PRICE_EXTRA_PARAMS JSON: %s", e)

    params = {
        "limit": 1,
        "sort": f"{timestamp_field} desc",
        **price_params,
    }

    request_params = {}
    for key, value in params.items():
        if isinstance(value, (dict, list)):
            request_params[key] = json.dumps(value, ensure_ascii=True)
        else:
            request_params[key] = value

    url = f"{base_url.rstrip('/')}{price_endpoint}"
    try:
        response = requests.get(url, params=request_params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        logger.warning("Failed to query latest EDS price timestamp: %s", e)
        return None

    records = payload.get("records") if isinstance(payload, dict) else None
    if not records:
        return None

    record = records[0]
    candidates = [
        timestamp_field,
        "HourUTC",
        "HourDK",
        "Minutes5UTC",
        "Minutes5DK",
        "time",
        "DateTime",
        "datetime",
    ]

    value = None
    for candidate in candidates:
        if candidate in record:
            value = record[candidate]
            break

    if not value:
        return None

    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None

    return parsed.to_pydatetime()


def main() -> None:
    args = _parse_args()
    config = get_config()

    if args.end:
        end_time = pd.to_datetime(args.end, errors="coerce")
        if pd.isna(end_time):
            raise ValueError("Invalid --end timestamp")
        end_time = end_time.to_pydatetime()
    else:
        latest_timestamp = _get_latest_eds_timestamp()
        if latest_timestamp is not None:
            end_time = latest_timestamp
            logger.info(
                "Using latest available EDS price timestamp: %s", end_time.isoformat()
            )
        else:
            end_time = datetime.now(timezone.utc)

    if args.start:
        start_time = pd.to_datetime(args.start, errors="coerce")
        if pd.isna(start_time):
            raise ValueError("Invalid --start timestamp")
        start_time = start_time.to_pydatetime()
    else:
        lookback_hours = args.lookback_hours or int(
            os.getenv("INGEST_LOOKBACK_HOURS", "24")
        )
        start_time = end_time - timedelta(hours=lookback_hours)

    weather_base_url = os.getenv("WEATHER_BASE_URL", "")
    if weather_base_url and "forecast" in weather_base_url.lower():
        today = datetime.now(timezone.utc).date()
        if end_time.date() < today:
            os.environ["WEATHER_BASE_URL"] = (
                "https://archive-api.open-meteo.com/v1/archive"
            )
            logger.info("Switching to Open-Meteo archive for historical window")

    fetcher = _build_fetcher(config)
    validator = DataValidator(null_threshold=config.data.null_value_threshold)

    pipeline = DataIngestionPipeline(
        fetcher=fetcher,
        validator=validator,
        storage_path=config.data.raw_data_path,
    )

    df = pipeline.fetch_and_validate(start_date=start_time, end_date=end_time)
    if df is None:
        logger.error("Ingestion failed or validation errors")
        return

    pipeline.save_parquet(df, date=end_time)
    logger.info(
        "Ingestion completed for %s to %s", start_time.isoformat(), end_time.isoformat()
    )


if __name__ == "__main__":
    main()
