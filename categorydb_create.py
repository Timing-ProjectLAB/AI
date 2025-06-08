import json
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings

# 1. JSON 로드
with open("FINAL_key_cat.json", encoding="utf-8") as f:
    data = json.load(f)

# 2. category 추출
categories = set()
for item in data:
    if isinstance(item.get("category", ""), str):
        for cat in item["category"].split(","):
            categories.add(cat.strip())

# 3. 벡터 저장
docs = [Document(page_content=cat, metadata={"type": "category"}) for cat in categories]
embedding = OpenAIEmbeddings()
category_vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory="./categorydb"
)
category_vectordb.persist()
print(f"✅ {len(docs)} categories saved in ./categorydb")