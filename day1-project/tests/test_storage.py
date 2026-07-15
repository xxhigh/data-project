from pathlib import Path

import pandas as pd

import main
from main import (
    read_summary_csv,
    read_summary_parquet,
    summarize_country,
    summarize_ip_location,
    summarize_weather,
    write_summary_csv,
    write_summary_parquet,
)


def test_write_and_read_summary_csv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    summary = summarize_weather(
        main.WeatherData.model_validate(
            {
                "latitude": 37.55,
                "longitude": 127.0,
                "generationtime_ms": 0.12,
                "utc_offset_seconds": 32400,
                "timezone": "Asia/Seoul",
                "timezone_abbreviation": "GMT+9",
                "elevation": 34.0,
                "hourly_units": {
                    "temperature_2m": "°C",
                    "precipitation_probability": "%",
                },
                "hourly": {
                    "time": ["2026-07-15T00:00", "2026-07-15T01:00"],
                    "temperature_2m": [21.8, 30.8],
                    "precipitation_probability": [10, 100],
                },
            }
        )
    )

    path = write_summary_csv(summary)
    loaded = read_summary_csv("weather")

    assert path == tmp_path / "weather_summary.csv"
    assert loaded.loc[0, "temperature_range"] == "21.8°C ~ 30.8°C"


def test_write_and_read_summary_parquet(
    tmp_path: Path,
    monkeypatch,
    country_data,
) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    summary = summarize_country(country_data)

    path = write_summary_parquet(summary)
    loaded = read_summary_parquet("country")

    assert path == tmp_path / "country_summary.parquet"
    assert loaded.loc[0, "country"] == "Korea (Republic of)"
    assert loaded.loc[0, "languages"] == "Korean"


def test_each_summary_uses_its_own_file(
    tmp_path: Path, monkeypatch, country_data, ip_location_data
) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    summaries = [
        summarize_country(country_data),
        summarize_ip_location(ip_location_data),
    ]

    for summary in summaries:
        write_summary_csv(summary)

    csv_files = sorted(path.name for path in tmp_path.glob("*.csv"))

    assert csv_files == ["country_summary.csv", "location_summary.csv"]


def test_summary_to_dataframe_flattens_list_values(country_data) -> None:
    summary = summarize_country(country_data)
    dataframe = main.summary_to_dataframe(summary)

    assert isinstance(dataframe, pd.DataFrame)
    assert dataframe.loc[0, "currencies"] == "South Korean won"
