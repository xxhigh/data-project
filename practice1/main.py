# ==============================
# 작업 환경: Python 3.11
# 작성자: 김근홍
# 작성일: 2026-07-15
# 
# 설명: 자료구조 집계, 컴프리헨션, 제너레이터를 활용한 데이터 처리 실습 코드
#
# 업데이트 내용(최신순으로 정렬)
# ==============================

from ast import literal_eval
from collections import Counter, defaultdict
from pathlib import Path
from sys import getsizeof


# 파일 읽기 (경로 추출)
DATA_FILE = Path(__file__).with_name("Python_Practice1_Data.json")

# 데이터 파일 존재 여부 확인 함수
def check_data_file_exists(path):
    if not path.is_file():
        print(f"오류: 데이터 파일을 찾을 수 없습니다. ({path})")
        print("데이터 파일(Python_Practice1_Data.json)은 현재 실행하는 py 파일과 같은 폴더에 있어야 합니다.")
        raise SystemExit(1)


# sales 데이터 로드 함수
def load_sales_data(path):
    check_data_file_exists(path)

    # 현재 제공되는 데이터 파일은 JSON 형식이지만, 실제로는 Python의 dict/list 구조를 문자열로 저장한 형태입니다.
    # 따라서 literal_eval() 함수를 사용하여 문자열을 Python 객체로 변환합니다.
    text = path.read_text(encoding="utf-8")

    _, data_text = text.split("=", maxsplit=1)
    return literal_eval(data_text.strip())

# 거래량 1000 이상인 거래만 필터링하고 지역별 총합 계산 함수
def calculate_total_sales_local(sales):
    filtered_sales = [sale for sale in sales if sale["amount"] >= 1000] # 거래량 1000 이상인 거래만 필터링
    regions = dict.fromkeys(sale["region"] for sale in filtered_sales) # 지역 총합 계산을 위해 지역 목록 생성
    total_sales = {
        region: sum(sale["amount"] for sale in filtered_sales if sale["region"] == region)
        for region in regions
    }
    return total_sales

# 지역별 거래 건수 카운팅 함수 (내림차순 정렬)
def count_sales_by_region(sales):
    return Counter(sale["region"] for sale in sales).most_common() # Counter를 사용하여 지역별 거래 건수 카운팅 후 most_common()으로 내림차순 정렬


# 카테고리별 amount 리스트 생성 함수
def group_amounts_by_category(sales):
    category_amounts = defaultdict(list)
    for sale in sales:
        category_amounts[sale["category"]].append(sale["amount"])

    return category_amounts


# month, category 기준 총매출 계산 함수
def calculate_total_sales_by_month_category(sales):
    grouped_sales = defaultdict(list)
    for sale in sales:
        grouped_sales[(sale["month"], sale["category"])].append(sale["amount"])

    return {
        group: sum(amounts)
        for group, amounts in grouped_sales.items()
    }


# 제너레이터
def read_sales_rows(path):
    check_data_file_exists(path)

    with path.open(encoding="utf-8") as file:
        for line in file:
            row_text = line.strip()
            if not row_text or row_text.startswith("sales") or row_text == "]":
                continue

            if row_text.endswith(","):
                row_text = row_text[:-1]

            sale = literal_eval(row_text)
            if sale["amount"] >= 1000:
                yield sale

# 메모리 크기 포맷팅 함수
def format_bytes(size):
    return f"{size:,} bytes"


def main():
    # 리스트 버전: 모든 데이터를 메모리에 올린 뒤 처리
    sales_list = load_sales_data(DATA_FILE)
    list_size = getsizeof(sales_list)
    
    # 1번 실습
    print('========== 1번 실습 ==============')
    total_sales = calculate_total_sales_local(sales_list)
    print(total_sales)
    print('=================================\r\n')
    
    # 2번 실습
    region_counts = count_sales_by_region(sales_list)
    category_amounts = group_amounts_by_category(sales_list)

    print('========== 2번 실습 ==============')
    print(region_counts)
    print(category_amounts)
    print('=================================\r\n')

    # 3번 실습
    generator_size = getsizeof(read_sales_rows(DATA_FILE))
    
    print('========== 3번 실습 ==============')
    print("메모리 크기 비교")
    print(f"리스트 객체 크기: {format_bytes(list_size)}")
    print(f"제너레이터 객체 크기: {format_bytes(generator_size)}")
    print('=================================\r\n')

    # 4번 실습
    print('========== 4번 실습 ==============')
    month_category_sales = calculate_total_sales_by_month_category(sales_list)
    print(month_category_sales)
    print('=================================\r\n')


if __name__ == "__main__":
    main()