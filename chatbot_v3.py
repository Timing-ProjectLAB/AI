#!/usr/bin/env python3
# chatbot.py  ·  Adaptive Filtering + Keyword·Category Edition
# 실행: python3 chatbot.py
# 필요한 패키지: pip install langchain-openai langchain chromadb python-dotenv

import os, re, json
from typing import List, Tuple, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ─────────────────────────────────── #
# 0. 보조 함수 – 질의 재구성
# ─────────────────────────────────── #

def build_query(base_prompt: str,
                age: Optional[int],
                region: Optional[str],
                interests: Optional[List[str]]) -> str:
    """저장된 정보를 엮어 RAG용 자연어 질의 문자열 생성"""
    parts: List[str] = [base_prompt]
    if region:
        parts.append(f"{region} 거주")
    if age:
        parts.append(f"{age}세")
    if interests:
        parts.append(f"관심사 {', '.join(interests)}")
    return " ".join(parts)
# ─────────────────────────────────── #
# 1. 관심사 · 지역 맵
# ─────────────────────────────────── #
INTEREST_MAPPING = {
    "창업": ["창업", "스타트업", "기업 설립", "벤처", "소상공인", "사업", "자금지원"],
    "취업": ["취업", "일자리", "채용", "고용", "잡페어", "구직활동", "면접", "이력서", "자기소개서", "취업지원", "구직"],
    "운동": ["운동", "스포츠", "체육", "피트니스", "헬스", "헬스케어", "요가", "체육관"],
    "학업": ["학업", "학습", "공부", "교육", "학위", "대학생활", "대학", "연구"],
    "프로그램": ["프로그램", "워크숍", "세미나", "캠프", "연수", "교육프로그램", "훈련프로그램"],
    "장학금": ["장학금", "학비 지원", "등록금 지원", "교육비 지원", "학자금"],
    "해외연수": ["해외연수", "글로벌 연수", "교환학생", "어학연수", "해외교육"],
    "인턴십": ["인턴십", "현장실습", "산학협력", "인턴", "실무경험"],
    "주거": ["주거", "주택", "임대", "전세", "월세", "보증금", "부동산"],
    "복지": ["복지", "사회복지", "지원", "보조금", "바우처", "의료", "건강", "출산", "육아"],
    "참여": ["참여", "권리", "시민", "사회", "봉사", "활동", "동아리"],
    "직업교육": ["직업", "훈련", "기술", "자격증", "교육", "강좌", "직업훈련"],
    "해외취업": ["해외취업", "국외취업", "글로벌취업", "일자리", "진출"],
    "정신건강": ["정신건강", "상담", "심리", "스트레스", "우울증"],
    "금융지원": ["대출", "자금", "지원금", "보조금", "융자"]
}
REGION_KEYWORDS = {
    "서울": ["서울", "서울시"],
    "경기": ["경기", "경기도"],
    "인천": ["인천", "인천시"],
    "부산": ["부산"],
    "대구": ["대구"],
    "광주": ["광주"],
    "대전": ["대전"],
    "울산": ["울산"],
    "세종": ["세종"],
    "강원": ["강원", "강원도", "강원특별자치도"],
    "충북": ["충북", "충청북도"],
    "충남": ["충남", "충청남도"],
    "전북": ["전북", "전라북도"],
    "전남": ["전남", "전라남도"],
    "경북": ["경북", "경상북도"],
    "경남": ["경남", "경상남도"],
    "제주": ["제주", "제주도", "제주특별자치도"]
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
        "제주특별자치도 제주시", "제주특별자치도 서귀포시", "제주도"
    ]
}

# ─────────────────────────────────── #
# 2. 정책 키워드 · 카테고리
# ─────────────────────────────────── #
KEYWORDS = [
    "바우처", "해외진출", "장기미취업청년", "맞춤형상담서비스", "교육지원",
    "출산", "보조금", "중소기업", "벤처", "대출", "금리혜택",
    "인턴", "공공임대주택", "육아", "청년가장", "신용회복"
]

CATEGORIES = ["일자리", "복지문화", "참여권리", "교육", "주거"]

def extract_keywords(text: str) -> List[str]:
    """사전 키워드 + 한글 형태소 기반 간이 추출"""
    hits = [kw for kw in KEYWORDS if kw in text]
    # 보강: 2글자 이상 명사 빈도수 상위 5개 자동 추출(간단 regex)
    tokens = re.findall(r"[가-힣]{2,}", text)
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    sorted_extra = sorted((w for w in freq if w not in hits),
                          key=lambda w: freq[w],
                          reverse=True)[:5]
    return hits + sorted_extra

def extract_categories(cat_field: str) -> List[str]:
    if not cat_field:
        return []
    return [c.strip() for c in cat_field.split(",") if c.strip() in CATEGORIES]

# ─────────────────────────────────── #
# 3. 벡터스토어 로드/생성 (강화 버전)
# ─────────────────────────────────── #
def load_or_build_vectorstore(json_path: str,
                              persist_dir: str,
                              api_key: str) -> Chroma:
    """
    1) persist_dir 에 Chroma DB가 이미 있으면 그대로 로드
    2) 없으면 json_path(정책 리스트)를 읽어 Document 로 변환 후 Chroma DB 구축
    """
    os.environ["OPENAI_API_KEY"] = api_key
    embedding = OpenAIEmbeddings()

    # 1) 기존 DB 재사용
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        return Chroma(persist_directory=persist_dir,
                      embedding_function=embedding)

    # 2) JSON → Document 변환
    with open(json_path, encoding="utf-8") as f:
        policies = json.load(f)

    docs: List[Document] = []
    for p in policies:
        # (1) 본문 텍스트
        text = (
            f"정책명: {p['title']}\n"
            f"지원대상: "
            f"{int(p.get('min_age', 0))}세~{int(p.get('max_age', 99))}세 / "
            f"지역 {', '.join(p.get('region_name', []))}\n"
            f"소득 분위: {p.get('income_condition', '제한 없음')}\n"
            f"혜택: {p.get('support_content', '')}\n"
            f"신청방법: {p.get('apply_method', '')}\n"
            f"설명: {p.get('description', '')}\n"
            f"링크: {p.get('apply_url', '')}"
        )

        # (2) 키워드 병합: JSON 원본 + 자동 추출
        merged_keywords = list(set(
            p.get("keywords", []) + extract_keywords(text)
        ))

        # (3) 메타데이터
        docs.append(Document(
            page_content=text,
            metadata={
                "policy_id":        p.get("policy_id"),
                "title":            p["title"],
                "region":           ", ".join(p.get("region_name", [])),
                "categories":       extract_categories(p.get("category", "")),
                "keywords":         merged_keywords,
                "min_age":          int(p.get("min_age", 0)),
                "max_age":          int(p.get("max_age", 99)),
                "income_condition": p.get("income_condition", "제한 없음"),
                "apply_period":     p.get("apply_period", ""),   # YYYYMMDD~YYYYMMDD
            }
        ))

    # 3) 분할 & 임베딩
    splitter   = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    vectordb   = Chroma(persist_directory=persist_dir,
                        embedding_function=embedding)
    vectordb.add_documents(split_docs)
    vectordb.persist()
    return vectordb

# ─────────────────────────────────── #
# 4. 사용자 입력 파싱
# ─────────────────────────────────── #
from typing import Tuple, Optional, List

def parse_user_input(text: str) -> Tuple[Optional[int], Optional[str], Optional[List[str]]]:
    age = None
    if m := re.search(r"(?:만\s*)?(\d{2})\s*(?:세|살)", text):
        age = int(m.group(1))

    region = None
    for std_r, keywords in REGION_KEYWORDS.items():
        if any(k in text for k in keywords):
            region = std_r
            break

    interests = None
    matches = [std_i for std_i, kws in INTEREST_MAPPING.items() if any(k in text for k in kws)]
    if matches:
        interests = matches

    return age, region, interests

# 5. 정보 누락 확인 함수 추가
# ─────────────────────────────────── #
def missing_info(age, region, interests) -> List[str]:
    needs = []
    if age is None:
        needs.append("나이")
    if region is None:
        needs.append("지역")
    if not interests or len(interests) == 0:
        needs.append("관심사")
    return needs


def classify_user_type(text: str) -> str:
    known = ["청년내일채움공제", "도약계좌", "구직활동지원금", "국민취업지원제도", "정책명"]
    return "policy_expert" if any(kw in text for kw in known) else "policy_novice"
# ─────────────────────────────────── #


# ─────────────────────────────────── #
# 5. 시스템 프롬프트
# ─────────────────────────────────── #
SYSTEM = SystemMessagePromptTemplate.from_template("""
[ROLE]
당신은 대한민국 만 19~39세 청년을 위한 정책 안내 챗봇입니다. 사용자의 입력과 제공된 context 문서를 바탕으로, 해당 청년에게 가장 적합한 정책을 찾아 안내하는 역할을 수행합니다.
추천할 정책은 사용자의 나이, 지역, 관심사와 일치해야 합니다. 사용자의 나이가 정책의 지원대상 나이 범위 내에 있어야 하며, 지역과 관심사도 일치하는 정책을 우선적으로 추천하세요.
[TASK]
1. 사용자 입력에서 나이, 지역, 관심사를 파싱합니다.
2. 조건에 부합하는 정책을 최소 3건 추천합니다.
   - 조건이 불충분하면 조회량 상위 3건을 제안합니다.
3. 각 추천에 대해 근거 문서의 ID 또는 URL을 함께 제공합니다.
4. 각 정책은 소득 분위 조건을 함께 표시하세요.

[OUTPUT FORMAT - MARKDOWN]
- **정책명** (소득: ○○ : 지원내용 요약 (출처: 문서ID)
- **정책명** (소득: ○○ : 지원내용 요약 (출처: 문서ID)
- **정책명** (소득: ○○ : 지원내용 요약 (출처: 문서ID)

[EXCEPTIONS]
- 정책 미발견 시:  
  “해당 조건에 맞는 정책이 없습니다. 대신 전국 공통 정책 3건을 추천합니다.”  
- 입력 정보 부족 시:  
  “○○ 정보를 추가로 알려주시면 더 정확한 추천이 가능합니다.”

[EXAMPLE - NORMAL]
- **청년내일채움공제**: 중소기업 근무 청년에게 목돈 마련 지원 (출처: policy_123)
- **국민취업지원제도**: 취업준비 중 청년에게 맞춤형 취업지원 (출처: policy_456)
- **청년구직활동지원금**: 구직활동비 월 최대 50만원 지원 (출처: policy_789)

[EXAMPLE - FALLBACK]
해당 조건에 맞는 정책이 없습니다. 대신 전국 공통 정책 3건을 추천합니다.

[EXAMPLE - ASK INFO]
나이 또는 지역 정보를 알려주시면 더욱 정확한 추천이 가능합니다.
""")

combine_prompt = ChatPromptTemplate.from_messages([
    SYSTEM,
    HumanMessagePromptTemplate.from_template(
        "context:\n{context}\n\n질문: {question}\n\n한국어로 간결하게 답변하세요."
    ),
])

# ─────────────────────────────────── #
# 6. RAG 체인
# ─────────────────────────────────── #
def create_rag_chain(vectordb: Chroma, api_key: str) -> ConversationalRetrievalChain:
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",  
        return_messages=True
    )
    chain =  ConversationalRetrievalChain.from_llm(
        llm, retriever, memory=memory,
        combine_docs_chain_kwargs={"prompt": combine_prompt, "document_variable_name": "context"},
        output_key="answer", return_source_documents=True
    )
    return chain, llm

# ─────────────────────────────────── #
# 7. 가중치 필터 & 폴백
# ─────────────────────────────────── #
# 선형 가중합 모델 기반 필터링
# 가중치: 지역 0.5, 관심사 0.3, 키워드 0.2
# 모든 부분 점수는 0~1 사이로 정규화
MIN_SCORE = 0.4  # 임계값(총합 1.0 중 0.4 이상이면 채택)

W_REGION   = 0.5
W_INTEREST = 0.3
W_KEYWORD  = 0.2


def jaccard_similarity(a: set, b: set) -> float:
    """두 집합의 자카드 유사도(0~1). 공집합이면 0."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def filter_docs(docs,user_age: int, user_text: str, region: str, interests: List[str]):
    """
    docs        : LangChain Document 리스트
    user_age    : 나이 조건
    user_text   : 사용자가 입력한 원문
    region      : 파싱된 표준 지역(예: '서울')
    interests   : 파싱된 관심사 리스트(예: ['창업', '주거'])
    """
    filtered = []
    kw_hits = extract_keywords(user_text)          # 사용자 문장에서 추출된 키워드 집합
    interests_set = set(interests)

    for d in docs:
        # ─────────────────────── #
        # 0. 나이 필터 : 메타데이터가 없다면 통과
        # ─────────────────────── #
        min_age = d.metadata.get("min_age", 0)
        max_age = d.metadata.get("max_age",999)
        if user_age not in range(min_age, max_age + 1):
            continue

        # ─────────────────────── #
        # 1. 지역 점수 (R: 0 | 0.5 | 1)
        # ─────────────────────── #
        doc_region_str = d.metadata.get("region", "")
        if region and any(k in doc_region_str for k in REGION_MAPPING[region]):
            region_score = 1.0
        elif region and region in doc_region_str:          # 인접·광역 등 부분 일치
            region_score = 0.5
        else:
            region_score = 0.0

        # ─────────────────────── #
        # 2. 관심사 점수 (I: 0~1)
        # ─────────────────────── #
        policy_tags = set(d.metadata.get("categories", []))
        interest_score = jaccard_similarity(interests_set, policy_tags)

        # ─────────────────────── #
        # 3. 키워드 점수 (K: 0~1)
        # ─────────────────────── #
        if kw_hits:
            doc_keywords = set(d.metadata.get("keywords", []))
            keyword_score = len(doc_keywords & set(kw_hits)) / len(kw_hits)
        else:
            keyword_score = 0.0

        # ─────────────────────── #
        # 4. 최종 점수
        # ─────────────────────── #
        score = (
            W_REGION   * region_score +
            W_INTEREST * interest_score +
            W_KEYWORD  * keyword_score
        )

        if score >= MIN_SCORE:
            filtered.append((score, d))

    # 점수 높은 순 정렬 후 Document 리스트만 반환
    return [d for _, d in sorted(filtered, key=lambda x: x[0], reverse=True)]

# ─────────────────────────────────── #
# 9. 관심사 세부 분류 흐름 유도 (LLM 기반)
# ─────────────────────────────────── #
SUB_INTEREST_MAPPING = {
    "취업": {
        "면접준비": ["모의면접", "면접복장", "이력서 클리닉", "증명사진", "정장 대여"],
        "역량강화": ["직업훈련", "직무교육", "취업기술 향상", "잡케어", "자격증"],
        "현장경험": ["일 경험", "인턴십", "현장실습", "기업 연계 프로젝트"],
        "구직지원금": ["구직촉진수당", "취업성공수당", "취업장려금", "활동비 지원"],
        "고용연계": ["채용연계", "공공기관 채용", "청년채용 연계사업"]
    },
    "창업": {
        "멘토링·상담": ["창업상담", "창업컨설팅", "BM모델", "법률·회계", "세무지원"],
        "사업계획·기획": ["사업계획서 작성", "아이디어 고도화", "창업 R&D", "아이템 발굴"],
        "자금지원": ["금리지원", "보증금", "융자", "창업자금"],
        "창업교육": ["창업 교육", "창업포럼", "창업 아카데미", "네트워킹"]
    },
    "운동": {
        "건강관리": ["헬스케어", "건강검진", "건강서비스", "의료서비스"],
        "체육활동": ["피트니스", "요가", "스포츠센터", "체육관"],
        "정신건강": ["심리상담", "정서지원", "스트레스 완화", "우울증 지원"]
    },
    "주거": {
        "임대료지원": ["월세지원", "임대료 보조", "공공임대주택", "주거바우처"],
        "주택구입·대출": ["주택 대출", "전세 대출", "보증금 지원"],
        "주택개보수": ["주택정비", "리모델링", "빈집 활용"]
    }
}

# 세부 관심사 질문 유도 함수 (대화형 방식, 예시 동적 반영)
def prompt_sub_interest(main_interest: str) -> Optional[str]:
    sub_map = SUB_INTEREST_MAPPING.get(main_interest)
    if not sub_map:
        return None

    print(f"\nBot:\n{main_interest}과 관련해 아래와 같은 지원이 있어요:")
    suggestions = list(sub_map.keys())
    for idx, key in enumerate(suggestions, 1):
        example_keywords = ", ".join(sub_map[key][:2])
        print(f"- {key}: {example_keywords} 관련 지원")

    example_hint = ", ".join(suggestions[:2])
    print(f"\n특별히 궁금한 것이 있으신가요? (예: {example_hint} 등)")
    sel = input("관심 있는 내용을 적어주세요: ").strip()
    for key in suggestions:
        if key in sel:
            return key
    print("입력 내용을 바탕으로 특정 항목을 찾을 수 없었어요. 일반 추천을 진행할게요.")
    return None

# ─────────────────────────────────── #
# 8. 콘솔 채팅
# ─────────────────────────────────── #
def console_chat(chain, llm):
    print("(Ctrl+C 종료)\n")

    stored_age: Optional[int] = None
    stored_region: Optional[str] = None
    stored_interests: Optional[List[str]] = None

    while True:
        user = input("You: ").strip()
        if user.lower() in ["초기화", "reset", "처음"]:
            stored_age = stored_region = stored_interests = None
            print("\nBot:\n사용자 정보를 초기화했습니다. 다시 입력해 주세요.\n")
            continue

        user_type = classify_user_type(user)
        age, region, interests = parse_user_input(user)

        # 누적 저장: None이 아닌 경우만 업데이트
        if age is not None:
            stored_age = age
        if region is not None:
            stored_region = region
        if interests is not None:
            stored_interests = interests

        # 디버그
        print(f"[DEBUG] age={age}, region={region}, interests={interests}")
        print(f"[DEBUG] stored_age={stored_age}, stored_region={stored_region}, stored_interests={stored_interests}")

        # 세부 관심사 유도
        if stored_interests and len(stored_interests) == 1:
            main_interest = stored_interests[0]
            sub_interest = prompt_sub_interest(main_interest)
            if sub_interest:
                stored_interests = [sub_interest]
                print(f"\n[DEBUG] 세부 관심사 적용: {stored_interests}")

        # 정보 부족 시(초보자)
        if user_type == "policy_novice":
            needs = missing_info(stored_age, stored_region, stored_interests)
            if needs:
                print(f"\nBot:\n{' · '.join(needs)} 정보를 알려주시면 더 정확한 정책 추천이 가능합니다.\n")
                print("─" * 60)
                continue

        # 질의 재구성
        if user_type == "policy_expert":
            question_for_chain = user
            question_for_llm = user
        else:
            question_for_chain = build_query(
                base_prompt="청년 정책 추천 요청",
                age=stored_age,
                region=stored_region,
                interests=stored_interests
            )
            question_for_llm = f"사용자는 {stored_age}세, {stored_region} 거주, 관심사 {', '.join(stored_interests)}입니다. 이 조건에 맞는 정책을 추천해 주세요."

        # RAG 호출
        res = chain.invoke({"question": question_for_chain})

        if user_type == "policy_expert":
            docs = res["source_documents"]
        else:
            docs = filter_docs(
                res["source_documents"],
                user_age=stored_age,
                user_text=question_for_chain,
                region=stored_region,
                interests=stored_interests or []
            )
            if not docs:
                # 1차 벡터 검색
                fallback_docs = chain.retriever.vectorstore.similarity_search(question_for_chain, k=5)
                if stored_interests:
                    fallback_docs = [
                        d for d in fallback_docs
                        if any(c in d.metadata.get("categories", []) for c in stored_interests)
                    ]
                docs = fallback_docs[:3]

        # 답변
        if docs:
            context = "\n\n".join(d.page_content for d in docs)
            resp = llm.invoke(
                combine_prompt.format_prompt(context=context, question=question_for_llm).to_messages()
            )
            print(f"\nBot:\n{resp.content}\n")
        else:
            print("\nBot:\n정책 정보를 찾을 수 없습니다. 더 구체적으로 입력해 주세요.\n")

        print("─" * 60)
