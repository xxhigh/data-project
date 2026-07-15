# ==============================
# 작업 환경: Python 3.11
# 작성자: 김근홍
# 작성일: 2026-07-15
#
# 설명: Day1 종합 실습 - 실무형 수집, 검증, 품질 파이프라인
# API 호출, 데이터 검증, 요약, CSV/Parquet 저장 및 벤치마킹
#
# 업데이트 내용(최신순으로 정렬)
# ==============================


import asyncio
import cProfile
import json
import logging
import os
import pstats
import timeit
from pathlib import Path
from typing import Any, Callable

import httpx
import pandas as pd
from dotenv import load_dotenv
from memory_profiler import memory_usage
from pydantic import BaseModel, Field, ValidationError


class AppConfig(BaseModel):
    urls: list[str] = Field(default_factory=list)


# 날씨 API 응답 원본 데이터
class WeatherData(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    generationtime_ms: float = Field(ge=0)
    utc_offset_seconds: int = Field(ge=-43200, le=50400)
    timezone: str
    timezone_abbreviation: str
    elevation: float = Field(ge=-500, le=9000)
    hourly_units: dict[str, str]
    hourly: dict[str, list[Any]]

# 날씨 요약 데이터 모델
class WeatherSummary(BaseModel):
    location: str
    timezone: str
    elevation_m: float = Field(ge=-500, le=9000)
    forecast_hours: int = Field(ge=0)
    temperature_range: str
    max_precipitation_probability: str

# 국가 API 응답 원본 데이터
class CountryData(BaseModel):
    name: str
    alpha2Code: str = Field(min_length=2, max_length=2)
    alpha3Code: str = Field(min_length=3, max_length=3)
    capital: str | None = None
    region: str
    subregion: str | None = None
    population: int = Field(ge=0)
    demonym: str | None = None
    area: float = Field(ge=0)
    gini: float | None = Field(default=None, ge=0, le=100)
    timezones: list[str] = Field(default_factory=list)
    borders: list[str] = Field(default_factory=list)
    nativeName: str | None = None
    numericCode: str | None = None
    currencies: list[dict[str, Any]] = Field(default_factory=list)
    languages: list[dict[str, Any]] = Field(default_factory=list)
    flag: str | None = None

# 국가 요약 데이터 모델
class CountrySummary(BaseModel):
    country: str
    code: str
    capital: str
    region: str
    population: str
    area: str
    density_per_km2: float = Field(ge=0)
    currencies: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)

# IP 기반 위치 API 응답 원본 데이터
class IPBasedLocationData(BaseModel):
    status: str
    country: str
    countryCode: str = Field(min_length=2, max_length=2)
    region: str
    regionName: str
    city: str
    zip: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    timezone: str
    isp: str
    org: str
    as_: str = Field(alias="as")
    query: str

# IP 기반 위치 요약 데이터 모델
class LocationSummary(BaseModel):
    ip: str
    location: str
    coordinates: str
    timezone: str
    network: str

# 벤치마크 결과 데이터 모델
class BenchmarkResult(BaseModel):
    summary_name: str
    operation: str
    file_format: str
    path: str
    timeit_seconds: float = Field(ge=0)
    cprofile_seconds: float = Field(ge=0)
    memory_peak_mib: float = Field(ge=0)
    file_size_bytes: int = Field(ge=0)


Summary = WeatherSummary | CountrySummary | LocationSummary
OUTPUT_DIR = Path("output")
LOG_DIR = Path("Logs")
LOG_PATH = LOG_DIR / "app.log"
BENCHMARK_REPEAT = 5
logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    def __init__(self, data_name: str, errors: list[dict[str, Any]]) -> None:
        self.data_name = data_name
        self.errors = errors
        super().__init__(format_validation_errors(data_name, errors))


class ApiCallError(RuntimeError):
    def __init__(self, url: str, message: str) -> None:
        self.url = url
        self.message = message
        super().__init__(f"API call failed - url={url} - {message}")


# Pydantic 검증 오류를 읽기 쉬운 메시지로 변환하는 함수
def format_validation_errors(data_name: str, errors: list[dict[str, Any]]) -> str:
    messages = []
    for error in errors:
        field_path = ".".join(str(location) for location in error["loc"])
        messages.append(f"{field_path}: {error['msg']}")
    return f"{data_name} validation failed - " + "; ".join(messages)


# 로그를 콘솔과 파일에 동시에 저장하도록 설정하는 함수
def configure_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


# .env 파일에서 실행 설정을 읽어오는 함수
def load_config() -> AppConfig:
    load_dotenv()
    return AppConfig(urls=json.loads(os.getenv("URLS", "[]")))


# 단일 URL에 API 요청을 보내고 JSON 응답을 반환하는 함수
async def fetch_url(client: httpx.AsyncClient, url: str) -> dict[str, Any]:
    try:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as error:
        status_code = error.response.status_code
        raise ApiCallError(url, f"HTTP {status_code} response") from error
    except httpx.RequestError as error:
        raise ApiCallError(url, str(error)) from error


# 여러 URL에 대한 API 요청을 비동기로 병렬 실행하는 함수
async def fetch_all(urls: list[str]) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        return await asyncio.gather(*(fetch_url(client, url) for url in urls))


# 날씨 API 응답을 Pydantic 모델로 검증하고 변환하는 함수
def transform_weather_data(data: dict[str, Any]) -> WeatherData:
    try:
        return WeatherData.model_validate(data)
    except ValidationError as error:
        raise DataValidationError("weather", error.errors()) from error


# 국가 API 응답을 Pydantic 모델로 검증하고 변환하는 함수
def transform_country_data(data: dict[str, Any]) -> CountryData:
    try:
        return CountryData.model_validate(data)
    except ValidationError as error:
        raise DataValidationError("country", error.errors()) from error


# IP 기반 위치 API 응답을 Pydantic 모델로 검증하고 변환하는 함수
def transform_ip_based_location_data(data: dict[str, Any]) -> IPBasedLocationData:
    try:
        return IPBasedLocationData.model_validate(data)
    except ValidationError as error:
        raise DataValidationError("ip location", error.errors()) from error


# 검증된 날씨 데이터를 보기 좋은 요약 모델로 재가공하는 함수
def summarize_weather(data: WeatherData) -> WeatherSummary:
    temperatures = data.hourly.get("temperature_2m", [])
    precipitation = data.hourly.get("precipitation_probability", [])
    temperature_unit = data.hourly_units.get("temperature_2m", "")
    precipitation_unit = data.hourly_units.get("precipitation_probability", "%")

    min_temperature = min(temperatures) if temperatures else None
    max_temperature = max(temperatures) if temperatures else None
    max_precipitation = max(precipitation) if precipitation else None

    temperature_range = "N/A"
    if min_temperature is not None and max_temperature is not None:
        temperature_range = (
            f"{min_temperature:.1f}{temperature_unit} ~ "
            f"{max_temperature:.1f}{temperature_unit}"
        )

    max_precipitation_probability = "N/A"
    if max_precipitation is not None:
        max_precipitation_probability = f"{max_precipitation}{precipitation_unit}"

    return WeatherSummary(
        location=f"{data.latitude:.4f}, {data.longitude:.4f}",
        timezone=f"{data.timezone} ({data.timezone_abbreviation})",
        elevation_m=data.elevation,
        forecast_hours=len(data.hourly.get("time", [])),
        temperature_range=temperature_range,
        max_precipitation_probability=max_precipitation_probability,
    )


# 검증된 국가 데이터를 보기 좋은 요약 모델로 재가공하는 함수
def summarize_country(data: CountryData) -> CountrySummary:
    currency_names = [
        currency["name"]
        for currency in data.currencies
        if isinstance(currency.get("name"), str)
    ]
    language_names = [
        language["name"]
        for language in data.languages
        if isinstance(language.get("name"), str)
    ]

    return CountrySummary(
        country=data.name,
        code=f"{data.alpha2Code} / {data.alpha3Code}",
        capital=data.capital or "N/A",
        region=" / ".join(part for part in [data.region, data.subregion] if part),
        population=f"{data.population:,}",
        area=f"{data.area:,.0f} km2",
        density_per_km2=round(data.population / data.area, 2) if data.area else 0,
        currencies=currency_names,
        languages=language_names,
    )


# 검증된 IP 위치 데이터를 보기 좋은 요약 모델로 재가공하는 함수
def summarize_ip_location(data: IPBasedLocationData) -> LocationSummary:
    return LocationSummary(
        ip=data.query,
        location=", ".join(
            part for part in [data.city, data.regionName, data.country] if part
        ),
        coordinates=f"{data.lat:.4f}, {data.lon:.4f}",
        timezone=data.timezone,
        network=" / ".join(part for part in [data.isp, data.org, data.as_] if part),
    )


# 저장 가능한 형태로 리스트 값을 문자열로 변환하는 함수
def normalize_summary_value(value: Any) -> Any:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return value


# Summary 모델을 파일 저장용 dict 행으로 변환하는 함수
def summary_to_row(summary: Summary) -> dict[str, Any]:
    row = {
        key: normalize_summary_value(value)
        for key, value in summary.model_dump().items()
    }
    return row


# Summary 모델의 파일명에 사용할 이름을 구하는 함수
def get_summary_name(summary: Summary) -> str:
    return summary.__class__.__name__.removesuffix("Summary").lower()


# Summary 이름과 파일 형식으로 저장 경로를 생성하는 함수
def get_summary_path(summary_name: str, file_format: str) -> Path:
    return OUTPUT_DIR / f"{summary_name}_summary.{file_format}"


# Summary 모델을 pandas DataFrame으로 변환하는 함수
def summary_to_dataframe(summary: Summary) -> pd.DataFrame:
    return pd.DataFrame([summary_to_row(summary)])


# Summary 데이터를 CSV 파일로 저장하는 함수
def write_summary_csv(summary: Summary) -> Path:
    path = get_summary_path(get_summary_name(summary), "csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_to_dataframe(summary).to_csv(path, index=False)
    return path


# Summary CSV 파일을 읽어오는 함수
def read_summary_csv(summary_name: str) -> pd.DataFrame:
    path = get_summary_path(summary_name, "csv")
    return pd.read_csv(path).fillna("")


# Summary 데이터를 Parquet 파일로 저장하는 함수
def write_summary_parquet(summary: Summary) -> Path:
    path = get_summary_path(get_summary_name(summary), "parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_to_dataframe(summary).to_parquet(path, index=False)
    return path


# Summary Parquet 파일을 읽어오는 함수
def read_summary_parquet(summary_name: str) -> pd.DataFrame:
    path = get_summary_path(summary_name, "parquet")
    return pd.read_parquet(path).fillna("")


# 모든 Summary 데이터를 CSV와 Parquet 파일로 저장하는 함수
def write_summary_files(summaries: list[Summary]) -> list[Path]:
    output_paths = []
    for summary in summaries:
        output_paths.append(write_summary_csv(summary))
        output_paths.append(write_summary_parquet(summary))
    return output_paths


# cProfile로 단일 작업의 실행 시간을 측정하는 함수
def profile_operation(operation: Callable[[], Any]) -> float:
    profiler = cProfile.Profile()
    profiler.runcall(operation)
    return pstats.Stats(profiler).total_tt


# memory_profiler로 단일 작업의 피크 메모리를 측정하는 함수
def measure_memory_peak(operation: Callable[[], Any]) -> float:
    usage = memory_usage((operation, (), {}), interval=0.01, max_usage=True)
    return float(usage)


# 단일 파일 읽기/쓰기 작업의 시간과 메모리를 벤치마크하는 함수
def benchmark_operation(
    summary_name: str,
    operation_name: str,
    file_format: str,
    path: Path,
    operation: Callable[[], Any],
) -> BenchmarkResult:
    timeit_seconds = timeit.timeit(operation, number=BENCHMARK_REPEAT)
    cprofile_seconds = profile_operation(operation)
    memory_peak_mib = measure_memory_peak(operation)

    return BenchmarkResult(
        summary_name=summary_name,
        operation=operation_name,
        file_format=file_format,
        path=str(path),
        timeit_seconds=round(timeit_seconds / BENCHMARK_REPEAT, 6),
        cprofile_seconds=round(cprofile_seconds, 6),
        memory_peak_mib=round(memory_peak_mib, 3),
        file_size_bytes=path.stat().st_size if path.exists() else 0,
    )


# 단일 Summary에 대해 CSV와 Parquet 읽기/쓰기 성능을 측정하는 함수
def benchmark_summary_storage(summary: Summary) -> list[BenchmarkResult]:
    summary_name = get_summary_name(summary)
    csv_path = get_summary_path(summary_name, "csv")
    parquet_path = get_summary_path(summary_name, "parquet")

    write_summary_csv(summary)
    write_summary_parquet(summary)

    return [
        benchmark_operation(
            summary_name,
            "write",
            "csv",
            csv_path,
            lambda: write_summary_csv(summary),
        ),
        benchmark_operation(
            summary_name,
            "read",
            "csv",
            csv_path,
            lambda: read_summary_csv(summary_name),
        ),
        benchmark_operation(
            summary_name,
            "write",
            "parquet",
            parquet_path,
            lambda: write_summary_parquet(summary),
        ),
        benchmark_operation(
            summary_name,
            "read",
            "parquet",
            parquet_path,
            lambda: read_summary_parquet(summary_name),
        ),
    ]


# 모든 Summary의 CSV와 Parquet 저장 형식별 성능을 측정하는 함수
def benchmark_storage_formats(summaries: list[Summary]) -> pd.DataFrame:
    benchmark_results = []
    for summary in summaries:
        benchmark_results.extend(benchmark_summary_storage(summary))

    return pd.DataFrame(result.model_dump() for result in benchmark_results)


# 저장 형식별 성능 비교 결과를 로그로 출력하는 함수
def log_storage_comparison(benchmark_df: pd.DataFrame) -> None:
    columns = [
        "summary_name",
        "operation",
        "file_format",
        "timeit_seconds",
        "cprofile_seconds",
        "memory_peak_mib",
        "file_size_bytes",
        "path",
    ]
    benchmark_df = benchmark_df[columns].sort_values(
        ["summary_name", "operation", "file_format"]
    )

    logger.info(
        "Storage benchmark comparison:\n%s", benchmark_df.to_string(index=False)
    )


# 전체 API 수집, 검증, 요약, 저장 벤치마크 파이프라인을 실행하는 함수
async def main() -> None:
    configure_logging()
    config = load_config()
    try:
        results = await fetch_all(config.urls)
    except ApiCallError as error:
        logger.error("%s", error)
        return

    try:
        summaries = [
            summarize_weather(transform_weather_data(results[0])),
            summarize_country(transform_country_data(results[1])),
            summarize_ip_location(transform_ip_based_location_data(results[2])),
        ]
    except DataValidationError as error:
        logger.error("%s", error)
        return

    for summary in summaries:
        logger.info(
            "%s summary:\n%s",
            get_summary_name(summary),
            summary.model_dump_json(indent=2),
        )

    benchmark_df = benchmark_storage_formats(summaries)
    log_storage_comparison(benchmark_df)


if __name__ == "__main__":
    asyncio.run(main())
