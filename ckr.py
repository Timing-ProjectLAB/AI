import json
import pandas as pd

# 1. JSON 로드
with open('ms_v3_short.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. DataFrame으로 변환
df = pd.json_normalize(data)

# 3. explode로 리스트 형태 필드 평탄화
kw = df['keywords'].explode().dropna().unique().tolist()
cat = df['category'].dropna().unique().tolist()
reg = df['region_name'].explode().dropna().unique().tolist()

# 4. 출력
print("키워드 목록:", kw)
print("카테고리 목록:", cat)
print("지역 목록:", reg)