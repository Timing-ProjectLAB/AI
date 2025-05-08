#!/usr/bin/env python3
# chatbot.py  ·  Adaptive Filtering + Keyword·Category Edition
# 실행: python3 chatbot.py
# 필요한 패키지: pip install langchain-openai langchain chromadb python-dotenv

import os, re, json
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import (
    ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
)
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ─────────────────────────────────── #
# ① 관심사 · 지역 맵
# ─────────────────────────────────── #
INTEREST_MAPPING = {
    "창업":  ["창업", "스타트업", "기업 설립"],
    "취업":  ["취업", "일자리", "채용", "고용", "잡페어"],
    "운동":  ["운동", "스포츠", "체육", "피트니스", "헬스", "헬스케어"],
    "학업":  ["학업", "학습", "공부", "교육", "학위", "대학생활"],
    "프로그램": ["프로그램", "워크숍", "세미나", "캠프", "연수"],
    "장학금": ["장학금", "학비 지원", "등록금 지원", "교육비 지원"],
    "해외연수": ["해외연수", "글로벌 연수", "교환학생"],
    "인턴십": ["인턴십", "현장실습", "산학협력"]
}

REGION_MAPPING = {
    "서울": ["서울"],
    "경기": ["경기", "수도권"],
    "인천": ["인천"],
    "부산": ["부산"],
    "대구": ["대구"],
    "광주": ["광주"],
    "대전": ["대전"],
    "울산": ["울산"],
    "세종": ["세종"],
    "강원": ["강원"],
    "충북": ["충북"],
    "충남": ["충남"],
    "경북": ["경북"],
    "경남": ["경남"],
    "전북": ["전북"],
    "전남": ["전남"],
    "제주": ["제주"]
}

# ─────────────────────────────────── #
# ② 정책 키워드 · 카테고리
# ─────────────────────────────────── #
KEYWORDS = [
    "바우처", "해외진출", "장기미취업청년", "맞춤형상담서비스", "교육지원",
    "출산", "보조금", "중소기업", "벤처", "대출", "금리혜택",
    "인턴", "공공임대주택", "육아", "청년가장", "신용회복"
]

CATEGORIES = ["일자리", "복지문화", "참여권리", "교육", "주거"]

def extract_keywords(text: str) -> List[str]:
    """문장에서 KEYWORDS 중 포함된 항목 반환"""
    return [kw for kw in KEYWORDS if kw in text]

def extract_categories(cat_field: str) -> List[str]:
    """쉼표(,)로 구분된 카테고리 문자열 → 유효 카테고리 배열"""
    if not cat_field:
        return []
    parts = [c.strip() for c in cat_field.split(",")]
    return [p for p in parts if p in CATEGORIES]

# ─────────────────────────────────── #
# ③ 벡터스토어 로드/생성
# ─────────────────────────────────── #
def load_or_build_vectorstore(json_path: str, persist_dir: str, api_key: str) -> Chroma:
    os.environ["OPENAI_API_KEY"] = api_key
    embedding = OpenAIEmbeddings()

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embedding)

    with open(json_path, encoding="utf-8") as f:
        policies = json.load(f)

    docs: List[Document] = []
    for p in policies:
        text = (
            f"정책명: {p['title']}\n"
            f"지원대상: {p.get('min_age', 0)}세~{p.get('max_age', 99)}세 / 지역 {', '.join(p['region_name'])}\n"
            f"혜택: {p.get('support_content','')}\n"
            f"신청방법: {p.get('apply_method','')}\n"
            f"설명: {p.get('description','')}\n"
            f"링크: {p.get('apply_url','')}"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "region": ", ".join(p["region_name"]),
                "categories": extract_categories(p.get("category", "")),
                "keywords":   extract_keywords(text)
            }
        ))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    vectordb.add_documents(split_docs)
    vectordb.persist()
    return vectordb

# ─────────────────────────────────── #
# ④ 사용자 입력 파싱
# ─────────────────────────────────── #
def parse_user_input(text: str) -> Tuple[int, str, List[str]]:
    age = 0
    if m := re.search(r"(?:만\s*)?(\d{2})\s*(?:세|살)", text):
        age = int(m.group(1))

    region = ""
    for std_r, kws in REGION_MAPPING.items():
        if any(k in text for k in kws):
            region = std_r
            break

    interests = [
        std_i for std_i, kws in INTEREST_MAPPING.items() if any(k in text for k in kws)
    ]
    return age, region, interests

# ─────────────────────────────────── #
# ⑤ 시스템 프롬프트
# ─────────────────────────────────── #
SYSTEM = SystemMessagePromptTemplate.from_template("""
당신은 대한민국 청년을 위한 정책 안내 챗봇입니다.
사용자 조건과 context 문서를 바탕으로 조건에 맞는 정책을 제시하세요.
조건이 없으면 가장 유사한 정책을 3건까지 추천하세요.

출력 형식:
- 정책명: 지원내용
""")

combine_prompt = ChatPromptTemplate.from_messages([
    SYSTEM,
    HumanMessagePromptTemplate.from_template(
        "context:\n{context}\n\n질문: {question}\n\n한국어로 간결하게 답변하세요."
    ),
])

# ─────────────────────────────────── #
# ⑥ RAG 체인
# ─────────────────────────────────── #
def create_rag_chain(vectordb: Chroma, api_key: str) -> ConversationalRetrievalChain:
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",   # ✅ 저장할 필드 지정
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm, retriever, memory=memory,
        combine_docs_chain_kwargs={"prompt": combine_prompt, "document_variable_name": "context"},
        output_key="answer", return_source_documents=True
    )

# ─────────────────────────────────── #
# ⑦ 가중치 필터 & 폴백
# ─────────────────────────────────── #
MIN_SCORE = 1  # 필터 임계값

def filter_docs(docs, user_text: str, region: str, interests: List[str]):
    filtered = []
    kw_hits = extract_keywords(user_text)

    for d in docs:
        region_score = interest_score = keyword_score = 0

        # 지역 점수
        if region:
            if any(k in d.metadata.get("region", "") for k in REGION_MAPPING[region]):
                region_score = 1

        # 관심사 점수
        if interests:
            if any(
                kw in d.page_content
                for it in interests
                for kw in INTEREST_MAPPING[it]
            ):
                interest_score = 2 if len(interests) > 1 else 1

        # 키워드 점수
        if kw_hits and any(kw in d.metadata.get("keywords", []) for kw in kw_hits):
            keyword_score = 1

        score = region_score + interest_score + keyword_score
        if score >= MIN_SCORE:
            filtered.append((score, d))

    # 점수 높은 순
    return [d for _, d in sorted(filtered, key=lambda x: x[0], reverse=True)]

# ─────────────────────────────────── #
# ⑧ 콘솔 채팅
# ─────────────────────────────────── #
def console_chat(chain):
    print("(Ctrl+C 종료)\n")
    while True:
        user = input("You: ")
        age, region, interests = parse_user_input(user)
        res = chain.invoke({"question": user})

        docs = filter_docs(res["source_documents"], user, region, interests)

        if not docs:
            # 폴백: 유사도 top‑3 재검색
            docs = chain.retriever.vectorstore.similarity_search(user, k=3)
            print("\nBot:\n", res["answer"], "\n")
        else:
            print("\nBot:\n", res["answer"], "\n")

        print("─" * 60)

# ─────────────────────────────────── #
# ⑨ main
# ─────────────────────────────────── #
if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "")
    JSON_PATH   = "ms_v3_short.json"
    PERSIST_DIR = "./chroma_policies"

    vectordb  = load_or_build_vectorstore(JSON_PATH, PERSIST_DIR, api_key)
    rag_chain = create_rag_chain(vectordb, api_key)
    console_chat(rag_chain)