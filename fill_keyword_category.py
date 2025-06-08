import json
import pandas as pd
from tqdm import tqdm
import os
from dotenv import load_dotenv
from openai import OpenAI

# ───────────────────────────────────── #
# 1. API 키 설정
# ───────────────────────────────────── #
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    raise ValueError("❌ OpenAI API 키가 .env 파일에 설정되어 있지 않습니다.")

client = OpenAI(api_key=api_key)

# ───────────────────────────────────── #
# 2. GPT 기반 키워드 및 카테고리 추출
# ───────────────────────────────────── #
def generate_keywords_and_category(desc, support):
    prompt = f"""
다음은 청년 정책의 설명과 지원 내용입니다. 이 내용을 바탕으로 관련 핵심 키워드 5~10개를 쉼표로 구분하여 추출하고, 해당 정책을 분류할 수 있는 주요 카테고리를 최대 2개 제시하세요.

[설명]
{desc}

[지원 내용]
{support}

출력 형식:
keywords: 키워드1, 키워드2, ...
category: 카테고리1, 카테고리2
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 한국 청년 정책 분석 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
        )

        output = response.choices[0].message.content.strip()
        lines = output.split('\n')
        keywords, category = "", ""

        for line in lines:
            if "keywords" in line.lower():
                keywords = line.split(":", 1)[-1].strip()
            elif "category" in line.lower():
                category = line.split(":", 1)[-1].strip()

        return keywords, category

    except Exception as e:
        print(f"⚠️ GPT 처리 실패: {e}")
        return "", ""

# ───────────────────────────────────── #
# 3. JSON 파일 불러오기
# ───────────────────────────────────── #
with open("FINAL.json", encoding="utf-8") as f:
    data = json.load(f)

df = pd.json_normalize(data)

# ───────────────────────────────────── #
# 4. GPT 기반 정보 생성
# ───────────────────────────────────── #
results = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="🔍 GPT 키워드 생성 중"):
    desc = row.get("description", "")
    support = row.get("support_content", "")
    keywords, category = generate_keywords_and_category(desc, support)
    results.append((keywords, category))

# ───────────────────────────────────── #
# 5. 결과 병합 및 저장
# ───────────────────────────────────── #
df["keywords"] = [r[0] for r in results]
df["category"] = [r[1] for r in results]

with open("FINAL_key_cat.json", "w", encoding="utf-8") as f:
    json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)