import pytest

from main import (
    DataValidationError,
    transform_country_data,
    transform_ip_based_location_data,
    transform_weather_data,
)


def test_transform_weather_data_returns_model(weather_payload: dict) -> None:
    data = transform_weather_data(weather_payload)

    assert data.latitude == 37.55
    assert data.hourly["temperature_2m"] == [21.8, 30.8]


def test_transform_weather_data_raises_readable_validation_error(
    weather_payload: dict,
) -> None:
    weather_payload["latitude"] = "not-a-number"

    with pytest.raises(DataValidationError, match="weather validation failed"):
        transform_weather_data(weather_payload)


def test_transform_country_data_validates_country_code(
    country_payload: dict,
) -> None:
    country_payload["alpha2Code"] = "KOR"

    with pytest.raises(DataValidationError, match="alpha2Code"):
        transform_country_data(country_payload)


def test_transform_ip_location_data_validates_coordinate_range(
    ip_location_payload: dict,
) -> None:
    ip_location_payload["lat"] = 120

    with pytest.raises(DataValidationError, match="lat"):
        transform_ip_based_location_data(ip_location_payload)
