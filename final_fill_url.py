import json
import pandas as pd
import re
import urllib.parse
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

tqdm.pandas()

# ───────────────────────────── #
# 1. JSON 파일 로드
# ───────────────────────────── #
with open("all_policy_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)  # 리스트 형태

df = pd.DataFrame(data)

# ───────────────────────────── #
# 2. URL 추출 함수 정의
# ───────────────────────────── #
def extract_url_from_text(text):
    if pd.isna(text):
        return ""
    match = re.search(r"https?://[^\s)]+", text)
    return match.group(0) if match else ""

# ───────────────────────────── #
# 3. applyUrlAddr이 없을 경우, plcyAplyMthdCn에서 URL 추출
# ───────────────────────────── #
df["aplyUrlAddr"] = df.apply(
    lambda row: extract_url_from_text(row["plcyAplyMthdCn"]) if row.get("aplyUrlAddr", "") == "" else row["aplyUrlAddr"],
    axis=1
)

# ───────────────────────────── #
# 4. 그래도 비어있는 경우, Naver 검색으로 URL 추정
# ───────────────────────────── #
def search_policy_url(title):
    try:
        query = urllib.parse.quote(f"{title} 신청 사이트")
        url = f"https://search.naver.com/search.naver?query={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        for a in soup.select('a[href^="https://"]'):
            href = a['href']
            if any(domain in href for domain in ['go.kr', 'bokjiro.go.kr', 'jobaba.net', 'youth']):
                return href
    except:
        return ""
    return ""

df["aplyUrlAddr"] = df.progress_apply(
    lambda row: search_policy_url(row["plcyNm"]) if row.get("aplyUrlAddr", "") == "" and extract_url_from_text(row.get("plcyAplyMthdCn", "")) == "" else row["aplyUrlAddr"],
    axis=1
)

# ───────────────────────────── #
# 5. 결과 저장
# ───────────────────────────── #
with open("all_policy_data_fill_url.json", "w", encoding="utf-8") as f:
    json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)