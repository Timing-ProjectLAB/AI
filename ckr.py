import json
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 👉 한글 폰트 설정 (Mac 기준)
plt.rcParams['font.family'] = 'AppleGothic'  # macOS용
plt.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지

# JSON 파일 로드
with open("final_data.json", encoding="utf-8") as f:
    data = json.load(f)

# category, keywords 추출
categories = []
keywords = []

for item in data:
    cat = item.get("category")
    if cat:
        categories.append(cat)

    kw_list = item.get("keywords")
    if kw_list and isinstance(kw_list, list):
        keywords.extend(kw_list)

# 카운트
category_counter = Counter(categories)
keyword_counter = Counter(keywords)

# 상위 10개 추출
top_categories = category_counter.most_common(30)
top_keywords = keyword_counter.most_common(30)

cat_labels, cat_counts = zip(*top_categories)
kw_labels, kw_counts = zip(*top_keywords)

# 📊 Category 시각화
plt.figure(figsize=(10, 6))
plt.barh(cat_labels[::-1], cat_counts[::-1])
plt.title("상위 30개 Category")
plt.xlabel("빈도수")
plt.ylabel("Category")
plt.tight_layout()
plt.show()

# 📊 Keyword 시각화
plt.figure(figsize=(10, 6))
plt.barh(kw_labels[::-1], kw_counts[::-1])
plt.title("상위 30개 Keywords")
plt.xlabel("빈도수")
plt.ylabel("Keyword")
plt.tight_layout()
plt.show()