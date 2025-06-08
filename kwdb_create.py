# kwdb_create.py
import json
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings

# 1. JSON 파일 경로
json_path = "FINAL_key_cat.json"

# 2. JSON 로드 및 키워드 추출
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)

unique_keywords = set()
for item in data:
    kws = item.get("keywords", "")
    if isinstance(kws, str):
        unique_keywords.update([kw.strip() for kw in kws.split(",") if kw.strip()])

# 3. 문서로 변환
docs = [Document(page_content=kw, metadata={"type": "keyword"}) for kw in unique_keywords]

# 4. 임베딩 및 Chroma 저장
embedding = OpenAIEmbeddings()  # OpenAI API 키 환경변수 필요
persist_dir = "./kwdb"

vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory=persist_dir
)
vectordb.persist()

print(f"✅ {len(docs)}개의 키워드를 './kwdb'에 저장 완료했습니다.")