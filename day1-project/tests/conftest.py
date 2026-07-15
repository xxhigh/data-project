import pytest

from main import CountryData, IPBasedLocationData, WeatherData


@pytest.fixture
def weather_payload() -> dict:
    return {
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


@pytest.fixture
def country_payload() -> dict:
    return {
        "name": "Korea (Republic of)",
        "alpha2Code": "KR",
        "alpha3Code": "KOR",
        "capital": "Seoul",
        "region": "Asia",
        "subregion": "Eastern Asia",
        "population": 51780579,
        "demonym": "South Korean",
        "area": 100210.0,
        "gini": 31.4,
        "timezones": ["UTC+09:00"],
        "borders": ["PRK"],
        "nativeName": "대한민국",
        "numericCode": "410",
        "currencies": [{"code": "KRW", "name": "South Korean won"}],
        "languages": [{"iso639_1": "ko", "name": "Korean"}],
        "flag": "🇰🇷",
    }


@pytest.fixture
def ip_location_payload() -> dict:
    return {
        "status": "success",
        "country": "United States",
        "countryCode": "US",
        "region": "VA",
        "regionName": "Virginia",
        "city": "Ashburn",
        "zip": "20149",
        "lat": 39.03,
        "lon": -77.5,
        "timezone": "America/New_York",
        "isp": "Google LLC",
        "org": "Google Public DNS",
        "as": "AS15169 Google LLC",
        "query": "8.8.8.8",
    }


@pytest.fixture
def weather_data(weather_payload: dict) -> WeatherData:
    return WeatherData.model_validate(weather_payload)


@pytest.fixture
def country_data(country_payload: dict) -> CountryData:
    return CountryData.model_validate(country_payload)


@pytest.fixture
def ip_location_data(ip_location_payload: dict) -> IPBasedLocationData:
    return IPBasedLocationData.model_validate(ip_location_payload)
