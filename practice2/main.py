# ==============================
# 작업 환경: Python 3.11
# 작성자: 김근홍
# 작성일: 2026-07-15
# 
# 설명: 파일 I/O, 예외 처리, Pydantic 검증 파이프라인 실습 코드
#
# 업데이트 내용(최신순으로 정렬)
# ==============================

import csv
import json
from pathlib import Path
import logging
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# 설정
DATA_FILE = Path(__file__).with_name("Python_Practice2_Data.json")
VALID_CSV_FILE = Path(__file__).with_name("valid_sales.csv")
ERROR_JSON_FILE = Path(__file__).with_name("sales_errors.json")
LOG_FILE = Path(__file__).with_name("pipeline.log")

EXPECTED_VALID_COUNT = 4 # 검증 성공 레코드 수
EXPECTED_ERROR_COUNT = 3 # 검증 실패 레코드 수

# 오류 확인용 샘플 데이터
SAMPLE_DATA = [
    {"region": "서울", "category": "전자", "amount": 1500, "month": "2024-01"},
    {"region": "부산", "category": "의류", "amount": 0, "month": "2024-01"},
    {"region": "서울", "category": "의류", "amount": 1200, "month": "2024-02"},
    {"region": "대구", "category": "전자", "amount": 950, "month": "2024-01"},
    {"region": "", "category": "전자", "amount": 1350, "month": "2024-01"},
    {"region": "부산", "category": "전자", "amount": 0, "month": "2024-02"},
    {"region": "대전", "category": "식품", "amount": 1100, "month": "2024/03"},
]

# 로깅 설정
logger = logging.getLogger("pipeline")
logger.setLevel(logging.DEBUG)

# 로깅 파일 핸들러 설정
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logger.addHandler(file_handler)

class SalesRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    month: str = Field(min_length=1)
    region: str = Field(min_length=1)
    amount: int = Field(gt=0)
    category: str | None

# 데이터 파일 존재 여부 확인 함수
def check_data_file_exists(path):
    if not path.is_file():
        print(f"오류: 데이터 파일을 찾을 수 없습니다. ({path})")
        print("데이터 파일은 현재 실행하는 py 파일과 같은 폴더에 있어야 합니다.")
        raise SystemExit(1)

# 파일 읽기 (예외 처리 코드)
def safe_load_csv(path: str) -> list[dict]:
    check_data_file_exists(Path(path))

    try:
        text = Path(path).read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
            raise ValueError("데이터가 dict 리스트 형식이 아닙니다.")

        logger.info(f"Successfully loaded data from {path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        return []
    except (ValueError, SyntaxError) as e:
        logger.error(f"Data format error: {e}")
        return []
    finally:
        print("로딩 종료")

# 검증 파이프라인 반환: 성공 -> valid 리스트, 실패 -> errors 리스트
def validate_sales_records(data: list[dict]) -> tuple[list[SalesRecord], list[dict]]:
    valid, errors = [], []

    for row in data:
        try:
            valid.append(SalesRecord(**row))
        except ValidationError as e:
            errors.append({"row": row, "errors": e.errors()})
    return valid, errors

# 데이터를 CSV 파일로 저장하는 함수
def dump_csv(data: list[SalesRecord], path: str):
    logger.info(f"Dumping {len(data)} records to CSV")
    rows = [record.model_dump() for record in data] # Pydantic 모델을 dict로 변환

    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SalesRecord.model_fields.keys()) # CSV 파일 쓰기
        writer.writeheader()
        writer.writerows(rows)

# 데이터를 JSON 파일로 저장하는 함수
def dump_json(data: list[dict], path: str):
    logger.info(f"Dumping {len(data)} records to JSON")
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

# CSV 파일을 읽어오는 함수
def load_csv(path):
    check_data_file_exists(Path(path))

    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

# JSON 파일을 읽어오는 함수
def load_json(path: str):
    check_data_file_exists(Path(path))

    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    raw_data = safe_load_csv(DATA_FILE)
    assert raw_data != None
    print("safe load csv 통과 및 데이터 로딩 완료")

    raw_data = SAMPLE_DATA  # 테스트용 샘플 데이터 사용
    print("샘플 데이터 사용")

    valids, errors = validate_sales_records(raw_data)

    dump_csv(valids, VALID_CSV_FILE)
    dump_json(errors, ERROR_JSON_FILE)

    reloaded = load_csv(VALID_CSV_FILE)
    reloaded_errors = load_json(ERROR_JSON_FILE)

    assert len(reloaded) == EXPECTED_VALID_COUNT
    assert len(reloaded_errors) == EXPECTED_ERROR_COUNT

    logger.info(f"valid saved: {len(valids)}")
    logger.info(f"errors saved: {len(reloaded_errors)}")
    logger.info(f"reloaded valid count: {len(reloaded)}")

if __name__ == "__main__":
    main()
