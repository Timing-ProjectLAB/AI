from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Chroma 디렉터리 경로
persist_dir = "chroma_policies"  # 저장한 위치로 변경

# 임베딩 로드 (embedding_function은 동일해야 함)
embedding = OpenAIEmbeddings()

# 기존 저장된 VectorStore 불러오기
vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)
# 모든 벡터 정보를 가져오기
data = vectordb.get()

# 키 목록: ids, embeddings, documents, metadatas
print("저장된 문서 수:", len(data["ids"]))
print("첫 번째 문서 ID:", data["ids"][0])
