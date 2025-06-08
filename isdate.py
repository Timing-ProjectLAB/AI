import json
from datetime import datetime

# ───────────────────────────── #
# 1. 오늘 날짜 기준 설정
# ───────────────────────────── #
today = datetime.today().date()  # 오늘 날짜 사용
# today = datetime(2025, 6, 5).date()  # 테스트용 고정 날짜 사용 시

# ───────────────────────────── #
# 2. apply_period에서 종료일 추출
# ───────────────────────────── #
def parse_end_date(apply_period: str):
    try:
        if "~" in apply_period:
            end_str = apply_period.split("~")[1].strip()
            return datetime.strptime(end_str, "%Y%m%d").date()
    except ValueError:
        return None
    return None

# ───────────────────────────── #
# 3. JSON 데이터 필터링
# ───────────────────────────── #
def filter_expired_policies(policies: list) -> list:
    filtered = []
    for policy in policies:
        apply_period = policy.get("apply_period", "").strip()
        if not apply_period:
            filtered.append(policy)
            continue

        end_date = parse_end_date(apply_period)
        if end_date is None or end_date >= today:
            filtered.append(policy)
    return filtered

# ───────────────────────────── #
# 4. 파일 로딩 및 저장
# ───────────────────────────── #
def main(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        policies = json.load(f)

    filtered = filter_expired_policies(policies)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"유효한 정책 수: {len(filtered)}개")
    print(f"저장 경로: {output_file}")

# ───────────────────────────── #
# 5. 실행 경로 설정
# ───────────────────────────── #
if __name__ == "__main__":
    input_path = "final_data_url.json"
    output_path = "FINAL.json"
    main(input_path, output_path)