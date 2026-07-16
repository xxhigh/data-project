# ==============================
# 작업 환경: Python 3.11
# 작성자: 김근홍
# 작성일: 2026-07-16
# 
# 설명: 시각화 4종, 통계 검정, sklearn Pipeline 실습 코드
#
# 업데이트 내용(최신순으로 정렬)
# ==============================

import pandas as pd
import polars as pl
import duckdb
import timeit
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = "data/sales_100k.csv"
REQUIRED_COLUMNS = {'amount', 'region', 'category'}

#=== 데이터 시각화

# EDA 시각화 4종을 2x2 서브플롯으로 출력합니다.
# df: 데이터 프레임
def show_plot(df):
    validate_columns(df.columns)
    if 'order_date' not in df.columns:
        raise ValueError("월별 라인 차트를 만들기 위한 order_date 컬럼이 없습니다.")

    fig, axes = plt.subplots(2,2, figsize=(12,12))
    amount = pd.to_numeric(df['amount'], errors='coerce').dropna()
    if amount.empty:
        raise ValueError("amount 컬럼에 시각화할 수 있는 유효한 숫자 값이 없습니다.")

    plot_df = df.copy()
    plot_df['amount'] = pd.to_numeric(plot_df['amount'], errors='coerce')
    plot_df['order_date'] = pd.to_datetime(plot_df['order_date'], errors='coerce')

    # 1. amount 분포를 히스토그램과 KDE 곡선으로 확인합니다.
    axes[0, 0].hist(amount, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='white')
    if len(amount) > 1 and amount.std() > 0:
        sample = amount.sample(min(len(amount), 5000), random_state=42).to_numpy()
        x = np.linspace(amount.min(), amount.max(), 300)
        bandwidth = 1.06 * sample.std(ddof=1) * (len(sample) ** (-1 / 5))
        if bandwidth > 0:
            kde = np.exp(-0.5 * ((x[:, None] - sample[None, :]) / bandwidth) ** 2).mean(axis=1)
            kde = kde / (bandwidth * np.sqrt(2 * np.pi))
            axes[0, 0].plot(x, kde, color='crimson', linewidth=2)
    axes[0, 0].set_title('Amount Histogram + KDE')
    axes[0, 0].set_xlabel('Amount')
    axes[0, 0].set_ylabel('Density')

    # 2. amount의 이상치와 분포 범위를 박스플롯으로 확인합니다.
    axes[0, 1].boxplot(amount, vert=True, patch_artist=True, boxprops={'facecolor': 'lightgreen'})
    axes[0, 1].set_title('Amount Boxplot')
    axes[0, 1].set_ylabel('Amount')
    axes[0, 1].set_xticks([1])
    axes[0, 1].set_xticklabels(['amount'])

    # 3. 월별 총매출 추이를 라인 차트로 확인합니다.
    monthly_sales = (
        plot_df.dropna(subset=['order_date', 'amount'])
        .assign(month=lambda data: data['order_date'].dt.to_period('M').dt.to_timestamp())
        .groupby('month')['amount']
        .sum()
        .sort_index()
    )
    axes[1, 0].plot(monthly_sales.index, monthly_sales.values, marker='o', color='steelblue')
    axes[1, 0].set_title('Monthly Sales Trend')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Total Amount')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 4. 숫자형 컬럼 간 상관관계를 히트맵으로 확인합니다.
    corr = plot_df.select_dtypes(include='number').corr()
    im = axes[1, 1].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 1].set_title('Correlation Heatmap')
    axes[1, 1].set_xticks(range(len(corr.columns)))
    axes[1, 1].set_yticks(range(len(corr.columns)))
    axes[1, 1].set_xticklabels(corr.columns, rotation=45, ha='right')
    axes[1, 1].set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            axes[1, 1].text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    plt.show()

# 서울 부산 평균 매출 차이를 t-test로, 지역과 카테고리 독립성을 카이제곱 검정을 수행합니다.
# df: 데이터 프레임
# alpha: 기본 0.05 (유의미 여부 p-value < alpha)
def show_t_test_and_cross(df, alpha=0.05):
    validate_columns(df.columns)

    # 서울과 부산의 평균 매출 차이를 독립표본 t-test로 검정합니다.
    test_df = df.copy()
    test_df['amount'] = pd.to_numeric(test_df['amount'], errors='coerce')
    seoul_amount = test_df.loc[test_df['region'] == '서울', 'amount'].dropna()
    busan_amount = test_df.loc[test_df['region'] == '부산', 'amount'].dropna()

    if seoul_amount.empty or busan_amount.empty:
        raise ValueError("서울 또는 부산의 amount 유효값이 없어 t-test를 수행할 수 없습니다.")

    t_stat, p_value = stats.ttest_ind(seoul_amount, busan_amount, equal_var=False)
    print(f"\r\n서울 vs 부산 평균 매출 t-test: t통계량={t_stat:.3f}, p-value={p_value:.3f}")
    if p_value < alpha:
        print("해석: p-value가 0.05보다 작으므로 서울과 부산의 평균 매출 차이는 통계적으로 유의미합니다.")
    else:
        print("해석: p-value가 0.05 이상이므로 서울과 부산의 평균 매출 차이는 통계적으로 유의미하지 않습니다.")

    # 지역과 카테고리의 독립성을 카이제곱 검정으로 확인합니다.
    contingency_table = pd.crosstab(test_df['region'], test_df['category'])
    if contingency_table.empty:
        raise ValueError("지역과 카테고리 분할표가 비어 있어 카이제곱 검정을 수행할 수 없습니다.")

    chi2_stat, chi2_p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\r\n지역 x 카테고리 카이제곱 검정: p-value={chi2_p_value:.3f}")
    if chi2_p_value < alpha:
        print("해석: p-value가 0.05보다 작으므로 지역과 카테고리는 서로 독립이 아니라고 볼 수 있습니다.")
    else:
        print("해석: p-value가 0.05 이상이므로 지역과 카테고리는 서로 독립이라고 볼 수 있습니다.")


# sklearn ColumnTransformer와 Pipeline으로 전처리, 모델 학습, 평가, 저장, 재로딩을 수행합니다.
# df: 데이터 프레임
# model_path: 모델 저장 위치
def process_sklearn(df, model_path="sales_amount_pipeline.joblib"):
    validate_columns(df.columns)
    validate_amount_values(df)

    model_df = df.copy()
    model_df['amount'] = pd.to_numeric(model_df['amount'], errors='coerce')
    model_df = model_df.dropna(subset=['amount'])

    if 'order_date' in model_df.columns:
        order_date = pd.to_datetime(model_df['order_date'], errors='coerce')
        model_df['order_year'] = order_date.dt.year
        model_df['order_month'] = order_date.dt.month

    numeric_features = [
        column for column in ['quantity', 'unit_price', 'customer_age', 'order_year', 'order_month']
        if column in model_df.columns
    ]
    categorical_features = [
        column for column in ['region', 'category', 'payment_method', 'customer_gender']
        if column in model_df.columns
    ]

    if not numeric_features and not categorical_features:
        raise ValueError("sklearn 파이프라인에 사용할 feature 컬럼이 없습니다.")

    X = model_df[numeric_features + categorical_features]
    y = model_df['amount']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # 숫자 변환: 수량, 가격, 구매자 나이, 주문년도, 주문월
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    # 카테고리 변환: 지역, 분류, 구매방법, 구매자 성별
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    # 전처리
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ])

    # 파이프라인 생성
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression()),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    score = pipeline.score(X_test, y_test)
    mae = mean_absolute_error(y_test, predictions)

    print(f"sklearn Pipeline R2 score: {score:.4f}")
    print(f"sklearn Pipeline MAE: {mae:.2f}")

    joblib.dump(pipeline, model_path)
    print(f"모델 저장 완료: {model_path}")

    loaded_pipeline = joblib.load(model_path)
    loaded_predictions = loaded_pipeline.predict(X_test)
    loaded_score = loaded_pipeline.score(X_test, y_test)
    print(f"재로딩 모델 R2 score: {loaded_score:.4f}")
    print(f"재로딩 모델 예측 샘플: {loaded_predictions[:5]}")

    return loaded_pipeline

# 지역, 카테고리별 총매출을 Plotly Express 막대 차트로 만들고 HTML 파일로 저장합니다.
# df: 데이터 프레임
# output_path: 출력 저장 위치
def make_plotly(df, output_path="region_category_sales.html"):
    validate_columns(df.columns)
    validate_amount_values(df)

    plot_df = df.copy()
    plot_df['amount'] = pd.to_numeric(plot_df['amount'], errors='coerce')
    grouped_df = (
        plot_df.dropna(subset=['region', 'category', 'amount'])
        .groupby(['region', 'category'], as_index=False)
        .agg(total=('amount', 'sum'))
        .sort_values('total', ascending=False)
    )

    if grouped_df.empty:
        raise ValueError("Plotly 차트를 만들 수 있는 지역, 카테고리별 총매출 데이터가 없습니다.")

    fig = px.bar(
        grouped_df,
        x='region',
        y='total',
        color='category',
        barmode='group',
        title='지역, 카테고리별 총매출',
        labels={
            'region': '지역',
            'category': '카테고리',
            'total': '총매출',
        },
        hover_data={
            'region': True,
            'category': True,
            'total': ':,.0f',
        },
    )
    fig.update_layout(
        xaxis_title='지역',
        yaxis_title='총매출',
        legend_title_text='카테고리',
        template='plotly_white',
    )
    fig.write_html(output_path)
    print(f"Plotly HTML 저장 완료: {output_path}")
    return fig
    
#=== 



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
    print(f"결측치 수:\r\n{df.isna().sum()}")
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
    
    show_plot(df) # (EDA 시각화 4종)
    
    show_t_test_and_cross(df) # 통계 검정 (t-test, 카이제곱)

    process_sklearn(df) # sklearn Pipeline 구성, 학습, 평가, 저장, 재로딩
    
    plot_fig = make_plotly(df) # Plotly 인터랙티브 막대 차트 HTML 저장
    plot_fig.show()

if __name__ == "__main__":
    main()
