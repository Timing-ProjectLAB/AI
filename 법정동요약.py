import pandas as pd
import json
import chardet

# CSV 파일 경로
csv_file_path = '전국법정동.csv'

# 인코딩 감지
with open(csv_file_path, 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']

# CSV 읽기
df = pd.read_csv(csv_file_path, dtype=str, encoding=encoding)

# 폐지된 법정동 제거
df = df[df['삭제일자'].isna()]

# 앞 5자리 기준으로 매핑 생성
short_mapping = {}

for _, row in df.iterrows():
    code5 = row['법정동코드'][:5]
    region_name = ' '.join(
        str(val) for val in [row['시도명'], row['시군구명']] if pd.notna(val)
    )
    # 이미 있는 코드면 덮어쓰지 않음 (처음 등장한 값만 저장)
    if code5 not in short_mapping:
        short_mapping[code5] = region_name

# JSON 저장
with open('region_codes_short.json', 'w', encoding='utf-8') as f:
    json.dump(short_mapping, f, ensure_ascii=False, indent=2)

print("✅ 요약된 JSON 파일이 생성되었습니다: region_codes_short.json")