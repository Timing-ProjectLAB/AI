import json
import pandas as pd
import re
import urllib.parse
import requests
from bs4 import BeautifulSoup

# ───────────────────────────── #
# 1. JSON 파일 로드
# ───────────────────────────── #
with open("ms_v3_short.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.json_normalize(data)

# ───────────────────────────── #
# 2. apply_method에서 URL 추출
# ───────────────────────────── #
def extract_url(text):
    if pd.isna(text):
        return ""
    match = re.search(r"https?://[^\s)]+", text)
    return match.group(0) if match else ""

df["apply_url"] = df.apply(
    lambda row: extract_url(row["apply_method"]) if row["apply_url"] == "" else row["apply_url"],
    axis=1
)

# ───────────────────────────── #
# 3. Naver 검색 기반 자동 채움
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

df["apply_url"] = df.apply(
    lambda row: search_policy_url(row["title"]) if row["apply_url"] == "" and extract_url(row["apply_method"]) == "" else row["apply_url"],
    axis=1
)

# ───────────────────────────── #
# 4. 저장
# ───────────────────────────── #
with open("ms_v3_filled_urls.json", "w", encoding="utf-8") as f:
    json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)