# ==============================
# 작업 환경: Python 3.11
# 작성자: 김근홍
# 작성일: 2026-07-16
# 
# 설명: Pandas, Polars, DuckDB를 활용한 데이터 처리 및 분석 예제 코드
#
# 업데이트 내용(최신순으로 정렬)
# ==============================

import pandas as pd
import polars as pl
import duckdb
import timeit
from pathlib import Path

DATA_FILE = "data/sales_100k.csv"
REQUIRED_COLUMNS = {'amount', 'region', 'category'}


# CSV 파일이 실제로 존재하는지 확인합니다.
def validate_file(file_path):
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")


# 데이터 처리에 필요한 필수 컬럼이 모두 있는지 확인합니다.
def validate_columns(columns):
    missing_columns = REQUIRED_COLUMNS - set(columns)
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {sorted(missing_columns)}")


# amount 컬럼에 집계와 IQR 계산에 사용할 수 있는 유효한 숫자 값이 있는지 확인합니다.
def validate_amount_values(df):
    amount = pd.to_numeric(df['amount'], errors='coerce')
    if amount.notna().sum() == 0:
        raise ValueError("amount 컬럼에 유효한 숫자 값이 없습니다.")


# Polars LazyFrame에서 amount 컬럼의 유효한 숫자 값이 있는지 확인합니다.
def validate_amount_values_polars(lf):
    valid_count = lf.select(pl.col('amount').drop_nulls().count()).collect().item()
    if valid_count == 0:
        raise ValueError("amount 컬럼에 유효한 숫자 값이 없습니다.")


# DuckDB로 CSV의 필수 컬럼과 amount 유효값을 확인합니다.
def validate_duckdb_data(file_path):
    escaped_file_path = file_path.replace("'", "''")
    columns = duckdb.sql(
        f"DESCRIBE SELECT * FROM read_csv_auto('{escaped_file_path}')"
    ).df()['column_name']
    validate_columns(columns)

    valid_count = duckdb.sql(
        f"""
        SELECT COUNT(amount) AS valid_count
        FROM read_csv_auto('{escaped_file_path}')
        """
    ).fetchone()[0]
    if valid_count == 0:
        raise ValueError("amount 컬럼에 유효한 숫자 값이 없습니다.")


# Pandas를 사용하여 CSV 파일을 로드합니다.
def load_data_pandas(file_path):
    validate_file(file_path)
    return pd.read_csv(file_path)



# 기본 EDA를 수행합니다. (shape, info, describe)
def check_data_pandas(df):
    print(f"DataFrame shape: {df.shape}")
    print("DataFrame info:")
    df.info()
    print(df.describe())



# 이상치 제거를 수행합니다.
# IQR(Interquartile Range) 방법을 사용하여 이상치를 제거합니다.
# 이상치는 Q1 - 1.5 * IQR 보다 작거나 Q3 + 1.5 * IQR 보다 큰 값으로 정의됩니다.
def process_data_pandas(df, verbose=True):
    Q1 = df['amount'].quantile(0.25)
    Q3 = df['amount'].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    def_filtered = df[df['amount'].between(lo, hi)]
    if verbose:
        print(f'이상치 {(~df["amount"].between(lo,hi)).sum()}건 제거')
    return def_filtered



# name aggregation 을 수행합니다.
# 지역, 카테고리별로 그룹화하여 총합, 평균, 건수를 계산하고 총합 기준으로 내림차순 정렬합니다.
def groupby_data_pandas(df):
    result = (
        df.groupby(['region', 'category'])
        .agg(
            total=('amount', 'sum'),
            average=('amount', 'mean'),
            count=('amount', 'count'),
        )
        .sort_values('total', ascending=False)
    ).reset_index()
    return result


# Polars를 사용하여 CSV 파일을 로드하고 처리하는 함수
# Polars는 LazyFrame을 사용하여 데이터를 처리하며, IQR을 계산하여 이상치를 제거하고 그룹화 및 집계를 수행합니다.
def lazy_load_data_polars(file_path):
    validate_file(file_path)
    lf = pl.scan_csv(file_path, try_parse_dates=True) # scan csv
    validate_columns(lf.collect_schema().names())
    validate_amount_values_polars(lf)

    # 필터링을 위한 IQR 계산
    amount_stats = (
        lf.select(
            pl.col('amount').quantile(0.25).alias('q1'),
            pl.col('amount').quantile(0.75).alias('q3'),
        )
        .with_columns(
            iqr=pl.col('q3') - pl.col('q1'),
        )
        .with_columns(
            lo=pl.col('q1') - 1.5 * pl.col('iqr'),
            hi=pl.col('q3') + 1.5 * pl.col('iqr'),
        )
    )

    return (
        lf.join(amount_stats, how='cross') # 그룹화 전 IQR 계산 결과를 조인
        .filter(pl.col('amount').is_between(pl.col('lo'), pl.col('hi')))
        .filter(pl.col('region').is_not_null() & pl.col('category').is_not_null())
        .group_by(['region', 'category']) # 그룹화
        .agg( # 집계 aggregation
            pl.col('amount').sum().alias('total'),
            pl.col('amount').mean().alias('average'),
            pl.col('amount').count().alias('count'),
        )
        .sort('total', descending=True) # 정렬
        .collect()
    )



# DuckDB를 사용하여 SQL 쿼리를 실행합니다.
# 경고: 파일 경로에 따옴표가 들어가는 경우 문제가 발생할 수 있으므로, 파일 경로를 적절히 이스케이프해야 합니다.
def sql_duckdb(file_path):
    validate_file(file_path)
    validate_duckdb_data(file_path)
    escaped_file_path = file_path.replace("'", "''")
    query = """
        WITH sales AS (
            SELECT *
            FROM read_csv_auto('{file_path}')
        ),
        amount_stats AS (
            SELECT
                quantile_cont(amount, 0.25) AS q1,
                quantile_cont(amount, 0.75) AS q3
            FROM sales
        ),
        bounds AS (
            SELECT
                q1 - 1.5 * (q3 - q1) AS lo,
                q3 + 1.5 * (q3 - q1) AS hi
            FROM amount_stats
        )
        SELECT
            region,
            category,
            SUM(amount) AS total,
            AVG(amount) AS average,
            COUNT(amount) AS count
        FROM sales
        CROSS JOIN bounds
        WHERE amount BETWEEN lo AND hi
          AND region IS NOT NULL
          AND category IS NOT NULL
        GROUP BY region, category
        ORDER BY total DESC
    """.format(file_path=escaped_file_path)
    return duckdb.sql(query).df()



# 벤치마크용 Pandas 파이프라인을 실행하는 함수
def run_pandas_pipeline(file_path):
    df = load_data_pandas(file_path)
    validate_columns(df.columns)
    validate_amount_values(df)
    df = process_data_pandas(df, verbose=False)
    return groupby_data_pandas(df)



# 벤치마크를 수행하는 함수
# repeat_count: 각 도구별로 실행할 반복 횟수
# 각 도구별로 실행 시간을 측정하고 평균 실행 시간을 계산하여 결과를 반환합니다.
def benchmark_tools(file_path, repeat_count=10):
    benchmarks = {
        'pandas': lambda: run_pandas_pipeline(file_path),
        'polars': lambda: lazy_load_data_polars(file_path),
        'duckdb': lambda: sql_duckdb(file_path),
    }

    results = []
    for name, func in benchmarks.items():
        total_seconds = timeit.timeit(func, number=repeat_count)
        results.append({
            'tool': name,
            'repeat': repeat_count,
            'total_seconds': total_seconds,
            'avg_seconds': total_seconds / repeat_count,
        })

    return pd.DataFrame(results).sort_values('avg_seconds').reset_index(drop=True)



def main():
    df = load_data_pandas(DATA_FILE)
    validate_columns(df.columns)
    validate_amount_values(df)
    check_data_pandas(df)
    df = process_data_pandas(df)

    grouped_df = groupby_data_pandas(df)
    print(grouped_df)

    grouped_lf = lazy_load_data_polars(DATA_FILE)
    print(grouped_lf)

    grouped_duckdb = sql_duckdb(DATA_FILE)
    print(grouped_duckdb)

    benchmark_result = benchmark_tools(DATA_FILE, repeat_count=10)
    print(benchmark_result)



if __name__ == "__main__":
    main()
