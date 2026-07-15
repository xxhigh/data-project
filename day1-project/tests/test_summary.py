from main import (
    summarize_country,
    summarize_ip_location,
    summarize_weather,
)


def test_summarize_weather_reformats_forecast(weather_data) -> None:
    summary = summarize_weather(weather_data)

    assert summary.location == "37.5500, 127.0000"
    assert summary.forecast_hours == 2
    assert summary.temperature_range == "21.8°C ~ 30.8°C"
    assert summary.max_precipitation_probability == "100%"


def test_summarize_country_reformats_population_and_density(country_data) -> None:
    summary = summarize_country(country_data)

    assert summary.country == "Korea (Republic of)"
    assert summary.code == "KR / KOR"
    assert summary.population == "51,780,579"
    assert summary.area == "100,210 km2"
    assert summary.density_per_km2 == 516.72
    assert summary.currencies == ["South Korean won"]
    assert summary.languages == ["Korean"]


def test_summarize_ip_location_reformats_network(ip_location_data) -> None:
    summary = summarize_ip_location(ip_location_data)

    assert summary.ip == "8.8.8.8"
    assert summary.location == "Ashburn, Virginia, United States"
    assert summary.coordinates == "39.0300, -77.5000"
    assert summary.network == "Google LLC / Google Public DNS / AS15169 Google LLC"
