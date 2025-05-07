import pandas as pd
import json
import chardet

# CSV 파일 경로
csv_file_path = '전국법정동.csv'  # 다운로드한 CSV 파일의 경로로 변경하세요.

# 1. 인코딩 자동 감지
with open(csv_file_path, 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']
    print(f"감지된 인코딩: {encoding}")

# 2. CSV 파일 읽기
df = pd.read_csv(csv_file_path, dtype=str, encoding=encoding)

# 3. 유효한 법정동 코드만 필터링 (폐지일자가 없는 행)
df = df[df['삭제일자'].isna()]

# 4. 법정동 코드와 행정구역명 매핑 생성
region_mapping = {}
for _, row in df.iterrows():
    code = row['법정동코드']
    region_name = ' '.join(
        str(val) for val in [row['시도명'], row['시군구명'], row['읍면동명'], row['리명']] if pd.notna(val)
    )
    region_mapping[code] = region_name

# 5. JSON 파일로 저장
json_file_path = 'region_codes.json'
with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(region_mapping, json_file, ensure_ascii=False, indent=2)

print(f'✅ JSON 파일이 생성되었습니다: {json_file_path}')