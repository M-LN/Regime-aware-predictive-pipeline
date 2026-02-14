"""
Run data ingestion for a given time range.
"""

import argparse
from datetime import datetime, timedelta
import os
import json
import logging
import pandas as pd

from src.config import get_config
from src.ingestion.data_fetcher import (
    DataIngestionPipeline,
    DataValidator,
    MockEnergyDataFetcher,
    EnergiDataServiceFetcher,
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

    field_map_raw = os.getenv("EDS_FIELD_MAP")
    if field_map_raw:
        try:
            field_map = json.loads(field_map_raw)
        except Exception as e:
            logger.warning("Invalid EDS_FIELD_MAP JSON: %s", e)

    extra_params_raw = os.getenv("EDS_EXTRA_PARAMS")
    if extra_params_raw:
        try:
            extra_params = json.loads(extra_params_raw)
        except Exception as e:
            logger.warning("Invalid EDS_EXTRA_PARAMS JSON: %s", e)

    return EnergiDataServiceFetcher(
        base_url=base_url,
        api_key=os.getenv("EDS_API_KEY"),
        timeout=int(os.getenv("EDS_TIMEOUT", config.data.api_timeout)),
        energy_endpoint=os.getenv("EDS_ENERGY_ENDPOINT", "/energy"),
        weather_endpoint=os.getenv("EDS_WEATHER_ENDPOINT", "/weather"),
        price_endpoint=os.getenv("EDS_PRICE_ENDPOINT", "/prices"),
        timestamp_field=os.getenv("EDS_TIMESTAMP_FIELD", "timestamp"),
        start_param=os.getenv("EDS_START_PARAM", "start"),
        end_param=os.getenv("EDS_END_PARAM", "end"),
        field_map=field_map,
        extra_params=extra_params,
    )


def main() -> None:
    args = _parse_args()
    config = get_config()

    if args.end:
        end_time = pd.to_datetime(args.end, errors="coerce")
        if pd.isna(end_time):
            raise ValueError("Invalid --end timestamp")
        end_time = end_time.to_pydatetime()
    else:
        end_time = datetime.utcnow()

    if args.start:
        start_time = pd.to_datetime(args.start, errors="coerce")
        if pd.isna(start_time):
            raise ValueError("Invalid --start timestamp")
        start_time = start_time.to_pydatetime()
    else:
        lookback_hours = args.lookback_hours or int(os.getenv("INGEST_LOOKBACK_HOURS", "24"))
        start_time = end_time - timedelta(hours=lookback_hours)

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
    logger.info("Ingestion completed for %s to %s", start_time.isoformat(), end_time.isoformat())


if __name__ == "__main__":
    main()
