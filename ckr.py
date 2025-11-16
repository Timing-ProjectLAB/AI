import json
from collections import Counter
import matplotlib.pyplot as plt

# 👉 한글 폰트 설정 (Mac 기준)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# JSON 파일 로드
with open("FINAL_key_cat.json", encoding="utf-8") as f:
    data = json.load(f)

# category, keywords 추출
categories = []
keywords = []

for item in data:
    # ─ Category 처리 ─ #
    cat = item.get("category")
    if cat:
        if isinstance(cat, str):
            categories.extend([c.strip() for c in cat.split(",")])
        elif isinstance(cat, list):
            categories.extend(cat)

    # ─ Keyword 처리 ─ #
    kw_list = item.get("keywords")
    if kw_list:
        if isinstance(kw_list, str):
            keywords.extend([kw.strip() for kw in kw_list.split(",")])
        elif isinstance(kw_list, list):
            keywords.extend(kw_list)

# 중복 없이 고유한 카테고리 및 키워드 수
unique_categories = set(categories)
unique_keywords = set(keywords)

print(f"🧾 고유한 카테고리 개수: {len(unique_categories)}개")
print(f"🧾 고유한 키워드 개수: {len(unique_keywords)}개")

# 카운트
category_counter = Counter(categories)
keyword_counter = Counter(keywords)

# 상위 30개 추출
top_categories = category_counter.most_common(30)
top_keywords = keyword_counter.most_common(30)

# 📊 시각화
if top_categories:
    cat_labels, cat_counts = zip(*top_categories)
    plt.figure(figsize=(10, 6))
    plt.barh(cat_labels[::-1], cat_counts[::-1])
    plt.title("상위 30개 Category")
    plt.xlabel("빈도수")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.show()
else:
    print("📌 Category 데이터가 없습니다.")

if top_keywords:
    kw_labels, kw_counts = zip(*top_keywords)
    plt.figure(figsize=(10, 6))
    plt.barh(kw_labels[::-1], kw_counts[::-1])
    plt.title("상위 30개 Keywords")
    plt.xlabel("빈도수")
    plt.ylabel("Keyword")
    plt.tight_layout()
    plt.show()
else:
    print("📌 Keyword 데이터가 없습니다.")