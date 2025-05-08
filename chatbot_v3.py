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
    "서울": [
        "서울특별시 종로구", "서울특별시 중구", "서울특별시 용산구", "서울특별시 성동구",
        "서울특별시 광진구", "서울특별시 동대문구", "서울특별시 중랑구", "서울특별시 성북구",
        "서울특별시 강북구", "서울특별시 도봉구", "서울특별시 노원구", "서울특별시 은평구",
        "서울특별시 서대문구", "서울특별시 마포구", "서울특별시 양천구", "서울특별시 강서구",
        "서울특별시 구로구", "서울특별시 금천구", "서울특별시 영등포구", "서울특별시 동작구",
        "서울특별시 관악구", "서울특별시 서초구", "서울특별시 강남구", "서울특별시 송파구",
        "서울특별시 강동구"
    ],
    "경기": [
        "경기도 수원시장안구", "경기도 수원시권선구", "경기도 수원시팔달구", "경기도 수원시영통구",
        "경기도 성남시수정구", "경기도 성남시중원구", "경기도 성남시분당구", "경기도 의정부시",
        "경기도 안양시만안구", "경기도 안양시동안구", "경기도 부천시원미구", "경기도 부천시소사구",
        "경기도 부천시오정구", "경기도 광명시", "경기도 평택시", "경기도 동두천시",
        "경기도 안산시상록구", "경기도 안산시단원구", "경기도 고양시덕양구", "경기도 고양시일산동구",
        "경기도 고양시일산서구", "경기도 과천시", "경기도 구리시", "경기도 남양주시",
        "경기도 오산시", "경기도 시흥시", "경기도 군포시", "경기도 의왕시", "경기도 하남시",
        "경기도 용인시처인구", "경기도 용인시기흥구", "경기도 용인시수지구", "경기도 파주시",
        "경기도 이천시", "경기도 안성시", "경기도 김포시", "경기도 화성시", "경기도 광주시",
        "경기도 양주시", "경기도 포천시", "경기도 여주시", "경기도 연천군", "경기도 가평군",
        "경기도 양평군"
    ],
    "인천": [
        "인천광역시 중구", "인천광역시 동구", "인천광역시 미추홀구", "인천광역시 연수구",
        "인천광역시 남동구", "인천광역시 부평구", "인천광역시 계양구", "인천광역시 서구",
        "인천광역시 강화군", "인천광역시 옹진군"
    ],
    "부산": [
        "부산광역시 중구", "부산광역시 서구", "부산광역시 동구", "부산광역시 영도구",
        "부산광역시 부산진구", "부산광역시 동래구", "부산광역시 남구", "부산광역시 북구",
        "부산광역시 해운대구", "부산광역시 사하구", "부산광역시 금정구", "부산광역시 강서구",
        "부산광역시 연제구", "부산광역시 수영구", "부산광역시 사상구", "부산광역시 기장군"
    ],
    "대구": [
        "대구광역시 중구", "대구광역시 동구", "대구광역시 서구", "대구광역시 남구",
        "대구광역시 북구", "대구광역시 수성구", "대구광역시 달서구", "대구광역시 달성군",
        "대구광역시 군위군"
    ],
    "광주": [
        "광주광역시 동구", "광주광역시 서구", "광주광역시 남구", "광주광역시 북구", "광주광역시 광산구"
    ],
    "대전": [
        "대전광역시 동구", "대전광역시 중구", "대전광역시 서구", "대전광역시 유성구", "대전광역시 대덕구"
    ],
    "울산": [
        "울산광역시 중구", "울산광역시 남구", "울산광역시 동구", "울산광역시 북구", "울산광역시 울주군"
    ],
    "세종": [
        "세종특별자치시 세종시"
    ],
    "강원": [
        "강원특별자치도 춘천시", "강원특별자치도 원주시", "강원특별자치도 강릉시", "강원특별자치도 동해시",
        "강원특별자치도 태백시", "강원특별자치도 속초시", "강원특별자치도 삼척시", "강원특별자치도 홍천군",
        "강원특별자치도 횡성군", "강원특별자치도 영월군", "강원특별자치도 평창군", "강원특별자치도 정선군",
        "강원특별자치도 철원군", "강원특별자치도 화천군", "강원특별자치도 양구군", "강원특별자치도 인제군",
        "강원특별자치도 고성군", "강원특별자치도 양양군"
    ],
    "충북": [
        "충청북도 청주시상당구", "충청북도 청주시서원구", "충청북도 청주시흥덕구", "충청북도 청주시청원구",
        "충청북도 충주시", "충청북도 제천시", "충청북도 보은군", "충청북도 옥천군", "충청북도 영동군",
        "충청북도 증평군", "충청북도 진천군", "충청북도 괴산군", "충청북도 음성군", "충청북도 단양군"
    ],
    "충남": [
        "충청남도 천안시동남구", "충청남도 천안시서북구", "충청남도 공주시", "충청남도 보령시", "충청남도 아산시",
        "충청남도 서산시", "충청남도 논산시", "충청남도 계룡시", "충청남도 당진시", "충청남도 금산군",
        "충청남도 부여군", "충청남도 서천군", "충청남도 청양군", "충청남도 홍성군", "충청남도 예산군",
        "충청남도 태안군"
    ],
    "전북": [
        "전북특별자치도 전주시완산구", "전북특별자치도 전주시덕진구", "전북특별자치도 군산시", "전북특별자치도 익산시",
        "전북특별자치도 정읍시", "전북특별자치도 남원시", "전북특별자치도 김제시", "전북특별자치도 완주군",
        "전북특별자치도 진안군", "전북특별자치도 무주군", "전북특별자치도 장수군", "전북특별자치도 임실군",
        "전북특별자치도 순창군", "전북특별자치도 고창군", "전북특별자치도 부안군"
    ],
    "전남": [
        "전라남도 목포시", "전라남도 여수시", "전라남도 순천시", "전라남도 나주시", "전라남도 광양시",
        "전라남도 담양군", "전라남도 곡성군", "전라남도 구례군", "전라남도 고흥군", "전라남도 보성군",
        "전라남도 화순군", "전라남도 장흥군", "전라남도 강진군", "전라남도 해남군", "전라남도 영암군",
        "전라남도 무안군", "전라남도 함평군", "전라남도 영광군", "전라남도 장성군", "전라남도 완도군",
        "전라남도 진도군", "전라남도 신안군"
    ],
    "경북": [
        "경상북도 포항시남구", "경상북도 포항시북구", "경상북도 경주시", "경상북도 김천시", "경상북도 안동시",
        "경상북도 구미시", "경상북도 영주시", "경상북도 영천시", "경상북도 상주시", "경상북도 문경시",
        "경상북도 경산시", "경상북도 의성군", "경상북도 청송군", "경상북도 영양군", "경상북도 영덕군",
        "경상북도 청도군", "경상북도 고령군", "경상북도 성주군", "경상북도 칠곡군", "경상북도 예천군",
        "경상북도 봉화군", "경상북도 울진군", "경상북도 울릉군"
    ],
    "경남": [
        "경상남도 창원시의창구", "경상남도 창원시성산구", "경상남도 창원시마산합포구", "경상남도 창원시마산회원구",
        "경상남도 창원시진해구", "경상남도 진주시", "경상남도 통영시", "경상남도 사천시", "경상남도 김해시",
        "경상남도 밀양시", "경상남도 거제시", "경상남도 양산시", "경상남도 의령군", "경상남도 함안군",
        "경상남도 창녕군", "경상남도 고성군", "경상남도 남해군", "경상남도 하동군", "경상남도 산청군",
        "경상남도 함양군", "경상남도 거창군", "경상남도 합천군"
    ],
    "제주": [
        "제주특별자치도 제주시", "제주특별자치도 서귀포시"
    ]
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
당신은 대한민국 청년을 위한 정책 안내 챗봇입니다. 사용자의 입력과 제공된 context 문서를 바탕으로, 해당 청년에게 가장 적합한 정책을 찾아 안내하는 역할을 수행합니다.

다음 사항을 반드시 준수하세요:

1. 사용자의 조건(나이, 지역, 관심사 등)이 명시되어 있으면, 해당 조건을 바탕으로 정확히 일치하거나 유사한 정책을 찾아 추천하세요.
2. 조건이 없거나 불분명한 경우에는 전체 context 중 가장 대표적이거나 많이 조회된 정책을 기준으로 최대 3건까지 추천하세요.
3. 정책명과 간단하고 명확한 지원 내용을 요약하여 보여주되, 불필요한 설명은 생략하세요.
4. 가능한 경우 정책명은 따옴표 없이 **볼드 처리**, 지원내용은 한 문장으로 요약하여 전달하세요.
5. 출력은 아래와 같은 형식을 반드시 따르세요:

- **정책명1**: 지원내용 요약1
- **정책명2**: 지원내용 요약2
- **정책명3**: 지원내용 요약3

6. 사용자가 특정 키워드(예: 창업, 주거, 장학금 등)를 언급했을 경우, 그에 맞는 카테고리 정책을 우선적으로 고려하세요.
7. 지역명이 포함된 경우, 해당 지역에 적용 가능한 정책만 우선 고려하며, 전국 공통 정책도 함께 제안할 수 있습니다.

당신은 친절하지만 간결하게 응답해야 하며, 정책 정보 외의 추가적인 설명이나 인사말은 생략합니다.
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