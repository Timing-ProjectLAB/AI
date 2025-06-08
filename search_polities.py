# search_policies_fallback.py

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def search_policies(query: str, user_age: int, user_region: str, interests: list[str], db_path: str = "./chroma_policies"):
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=db_path, embedding_function=embedding)

    # 1. Full 조건 검색
    filters_full = {
        "$and": [
            {"region": {"$eq": user_region}},
            {"min_age": {"$lte": user_age}},
            {"max_age": {"$gte": user_age}},
            {"categories": {"$in": interests}},
        ]
    }

    results = vectordb.similarity_search(query, k=5, filter=filters_full)

    if results:
        print(f"\n✅ [정확 일치 결과] 나이 {user_age}, 지역 '{user_region}', 관심사 {interests}")
        for idx, doc in enumerate(results, 1):
            print_result(idx, doc)
        return

    # 2. 지역 조건 제외
    filters_no_region = {
        "$and": [
            {"min_age": {"$lte": user_age}},
            {"max_age": {"$gte": user_age}},
            {"categories": {"$in": interests}},
        ]
    }

    results = vectordb.similarity_search(query, k=3, filter=filters_no_region)
    if results:
        print(f"\n🔄 [지역 제외 유사 결과] 나이 {user_age}, 관심사 {interests}")
        for idx, doc in enumerate(results, 1):
            print_result(idx, doc)
        return

    # 3. 나이 조건 제외
    filters_no_age = {
        "$and": [
            {"region": {"$eq": user_region}},
            {"categories": {"$in": interests}},
        ]
    }

    results = vectordb.similarity_search(query, k=3, filter=filters_no_age)
    if results:
        print(f"\n🔄 [나이 제외 유사 결과] 지역 '{user_region}', 관심사 {interests}")
        for idx, doc in enumerate(results, 1):
            print_result(idx, doc)
        return

    # 4. 관심사만 사용
    filters_keywords_only = {
        "categories": {"$in": interests}
    }

    results = vectordb.similarity_search(query, k=3, filter=filters_keywords_only)
    if results:
        print(f"\n🔄 [관심사 기반 일반 추천]")
        for idx, doc in enumerate(results, 1):
            print_result(idx, doc)
        return

    # 5. fallback - 필터 없이 검색
    print("\n🚫 조건에 맞는 정책은 없지만, 유사한 정책을 아래에 추천합니다.")
    results = vectordb.similarity_search(query, k=3)
    for idx, doc in enumerate(results, 1):
        print_result(idx, doc)

def print_result(idx, doc):
    print(f"\n📄 결과 {idx}")
    print("🧷 메타데이터:", doc.metadata)
    print("📝 내용 요약:", doc.page_content[:300], "...")


if __name__ == "__main__":
    search_policies(
        query="대출",
        user_age=25,
        user_region="서울",
        interests=["학자금 대출", "저소득 대학생", "대학 재학생"]
    )