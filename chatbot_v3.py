# 추가: 코드 맨 위에 clean_text_for_matching 함수 정의
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
from tqdm import tqdm
from functools import lru_cache
import time

# ─────────────────────────────────── #
# 📌 In‑memory session store (user‑level)
# ─────────────────────────────────── #
from collections import defaultdict
SESSION_STORE = defaultdict(lambda: {"user_info": None, "recommended_ids": set()})

# 특수문자 제거, 소문자화 등을 통해 키워드 매칭에 방해가 되는 요소들을 제거하는 함수
def clean_text_for_matching(text):
    return re.sub(r"[^\w\s]", "", text).replace("에", "").replace("에서", "").replace("인데", "").replace("야", "").strip()
# 사용자가 “다른 정책”, “추가로 보여줘” 같은 추가 추천 요청인지 판별하는 함수
def is_generic_more_request(text: str) -> bool:
    """
    사용자가 '다른 정책', '추가 정책', '더 보여줘' 등
    구체적 조건 없는 추가 추천을 요구하는지 판별.
    """
    text = text.strip()
    if re.search(r"다른\s*정책", text):
        return True
    if "정책" in text and re.search(r"(더|추가|또|없어|있어)", text):
        return True
    # "더 알려줘", "더 보여줘", "더 추천해줘" 처리
    if re.search(r"더\s*(알려줘|보여줘|추천해줘)", text):
        return True
    return False

# ─────────────────────────────────── #
# 정책 관련 질문 여부 판별 함수
# ─────────────────────────────────── #
NON_POLICY_KEYWORDS = [
    "안녕", "하이", "반가워", "잘 지냈어", "뭐해", "심심해", "놀자", "지루해", "고마워", "감사", "잘자", "잘 자", "굿밤",
    "누구야", "너 뭐야", "정체가 뭐야", "자기소개", "이름", "챗지피티", "gpt", "ai야", "로봇이야", "몇 살", "나이",
    "날씨", "온도", "기온", "몇 시", "시간", "오늘 날짜", "지금 몇시", "오늘 뭐야", "요일",
    "기분 어때", "사랑해", "귀여워", "좋아해", "여자친구", "남자친구", "썸", "연애", "이상형",
    "퀴즈", "수수께끼", "농담", "웃겨줘", "재밌는 얘기", "우주", "과학", "역사", "유튜브", "게임", "유머"
]
# 간단한 휴리스틱과 키워드 리스트(NON_POLICY_KEYWORDS)를 기반으로, 입력이 정책 관련 질의인지 1차 검사하는 함수
def is_policy_related_question(text: str) -> bool:
    import re
    # uses global clean_text_for_matching and REVERSE_REGION_LOOKUP
    if len(text.strip()) < 2:
        return False
    cleaned = re.sub(r"[ㅋㅎㅠㅜ]+", "", text.lower())
    for word in NON_POLICY_KEYWORDS:
        if word in cleaned:
            return False
    if re.match(r"^[가-힣]{1,3}(야|이야)?$", text.strip()):
        # Clean conversational endings like '야', '이야'
        clean_key = clean_text_for_matching(text)
        # If cleaned key matches a region in lookup, treat as policy-related (region input)
        if clean_key in REVERSE_REGION_LOOKUP:
            return True
        return False
    return True

# ─────────────────────────────────── #
# LLM 기반 정책 질문 여부 판별 함수
# ─────────────────────────────────── #
# 위 휴리스틱이 모호할 때, GPT-4o-mini에 “Y/N” 분류를 요청해 보다 정확히 판단하고, 실패 시 rule-based로 폴백
@lru_cache(maxsize=1024)           # 같은 문장은 한 번만 문의
def is_policy_related_question_llm(text: str) -> bool:
    """
    GPT-4o-mini로 ‘정책 관련 질문인지’ Y/N 분류.
    - refined heuristic for short/numeric/keyword input
    - LLM 오류 시 rule-based 폴백.
    """
    cleaned = text.strip()
    if not cleaned:
        return False  # 빈 입력

    # '다른 정책', '추가 정책' 등 일반 추가 추천 요청은 정책 관련으로 간주
    if is_generic_more_request(cleaned):
        return True

    # ① 숫자 1~2자리만 입력 → 나이로 간주 → 정책 질문 True
    if re.fullmatch(r"\d{1,2}", cleaned):
        return True

    # ② 단어 1~2자여도 관심사·지역 키워드라면 True
    if cleaned in REVERSE_REGION_LOOKUP:
        return True
    if cleaned in INTEREST_MAPPING:
        return True
    if any(cleaned in kws for kws in INTEREST_MAPPING.values()):
        return True

    # ③ 1글자·특수문자·웃음(ㅋㅎ)만 → False
    if len(cleaned) == 1 or re.fullmatch(r"[ㅋㅎ]+", cleaned):
        return False

    # ④ '정책' 명시, 나이·지역·관심 표현이 있으면 즉시 정책 관련으로 분류
    if "정책" in cleaned:
        return True
    if re.search(r"\d{1,2}\s*(세|살)", cleaned):
        return True
    if any(keyword in cleaned for keyword in ["나이", "지역", "관심"]):
        return True

    # ⑤ 파서가 핵심 조건을 추출하면 정책 맥락으로 간주
    parsed_age, parsed_region, parsed_interests = parse_user_input(text)
    if parsed_age or parsed_region or parsed_interests:
        return True

    system_msg = (
        "너는 대한민국 청년 정책 상담 챗봇의 분류기야. "
        "아래 사용자 입력이 정책과 *관련된 질문*인지 판단해. "
        "대답은 'Y' 또는 'N' 중 하나로만."
    )
    user_msg = f"사용자 입력: {cleaned}\n\n정책 관련 질문인가?"

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0,
            max_tokens=1,
        )
        content = resp.choices[0].message.content if resp.choices else ""
        return (content or "").strip().upper().startswith("Y")
    except Exception:
        # 네트워크/쿼터 문제 시 휴리스틱으로 폴백
        return is_policy_related_question(cleaned)

# 토큰 수나 핵심 키워드 포함 여부를 보고 “유효한 정책 질의”인지 추가 검사하는 함수
def is_valid_query(text: str) -> bool:
    # 숫자만 입력되어도 나이로 간주
    if re.search(r"\b\d{1,2}\b", text):
        return True

    tokens = re.findall(r"[가-힣a-zA-Z0-9]+", text)

    # 1) 3단어 이상이면 무조건 정책 관련 질의로 간주
    if len(tokens) >= 3:
        return True

    # 2) 2단어짜리 짧은 질문이라도 핵심 키워드가 포함되면 허용
    if len(tokens) == 2:
        for tok in tokens:
            if (
                tok == "정책" or
                tok in KEYWORDS or
                tok in INTEREST_MAPPING or
                any(tok in kws for kws in INTEREST_MAPPING.values())
            ):
                return True
    return False
# 지역명 키워드 여부 판별 함수
def is_region_keyword(word: str) -> bool:
    return word in REVERSE_REGION_LOOKUP or any(word in names for names in REGION_MAPPING.values())

# 사용자 입력에서 정보 자동 추출 함수
def extract_user_info(user_input: str):
    info = {
        "age": None,
        "region": None,
        "interests": [],
        "status": None,
        "income": None,
        "gender": None,
        "education": None,
        "desired_job": None,
        "hope_region": None,
        "family": None,
        "special": None,
    }

    # ✅ 전처리: 마침표, 쉼표 등 제거 → '여주에 사는 25살이야'로 만들기
    clean_text = re.sub(r"[^\w가-힣]", " ", user_input)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    # 🔁 정확한 나이/지역/관심사 파싱은 parse_user_input() 재활용
    parsed_age, parsed_region, parsed_interests = parse_user_input(clean_text)
    info["age"] = parsed_age
    info["region"] = parsed_region
    info["interests"] = parsed_interests if parsed_interests else []

    # 상태 추출
    if "대학생" in user_input:
        info["status"] = "대학생"
    elif "취준생" in user_input or "취업 준비" in user_input:
        info["status"] = "취업준비생"
    elif "졸업생" in user_input or "졸업 예정" in user_input:
        info["status"] = "졸업예정자"
    elif "재직" in user_input or "직장인" in user_input or "근무 중" in user_input:
        info["status"] = "재직자"
    elif "프리랜서" in user_input or "프리랜서" in clean_text:
        info["status"] = "프리랜서"

    # 소득
    if "저소득" in user_input:
        info["income"] = "저소득층"
    elif "고소득" in user_input:
        info["income"] = "고소득층"
    else:
        m_income = re.search(r"(?:중위\s*소득|소득)\s*(\d{2,3})\s*%", user_input)
        if m_income:
            info["income"] = f"중위소득 {m_income.group(1)}%"

    # 성별
    if "여자" in user_input or "여성" in user_input:
        info["gender"] = "여성"
    elif "남자" in user_input or "남성" in user_input:
        info["gender"] = "남성"

    # 학력/전공
    edu_match = re.search(r"([가-힣A-Za-z]+과)\s*(?:를)?\s*(?:전공|졸업|졸업할|졸업 예정)", user_input)
    if edu_match:
        info["education"] = f"{edu_match.group(1)} 전공"
    elif "졸업" in user_input:
        info["education"] = "졸업/졸업 예정"
    elif "재학" in user_input:
        info["education"] = "재학 중"

    # 희망 직무
    desired_jobs = ["개발자", "디자이너", "간호사", "엔지니어", "교사", "연구원", "마케터", "공무원", "데이터 분석가"]
    for job in desired_jobs:
        if job in user_input:
            info["desired_job"] = job
            break

    # 희망 근무 지역
    if any(keyword in user_input for keyword in ["근무", "일하고", "취업", "취직", "일자리"]):
        hoped_region = extract_region(user_input, REGION_MAPPING)
        if hoped_region:
            info["hope_region"] = hoped_region

    # 가족/부양 정보
    if "자녀" in user_input or "아이" in user_input or "육아" in user_input:
        info["family"] = "자녀 있음"
    elif "기혼" in user_input or "결혼" in user_input or "신혼" in user_input:
        info["family"] = "기혼"

    # 특별 대상
    special_keywords = ["장애", "국가유공", "보훈", "다문화", "군필", "전역"]
    for sk in special_keywords:
        if sk in user_input:
            info["special"] = sk
            break

    return info


def print_result(idx, doc):
    result = {
        "policy_id": doc.metadata.get("policy_id", f"unknown_{idx}"),
        "name":      doc.metadata.get("title"),
        "summary":   doc.metadata.get("summary"),
        "eligibility": f"{doc.metadata.get('min_age','?')}~{doc.metadata.get('max_age','?')}세 / {doc.metadata.get('region','전국')}",
        "period":    doc.metadata.get("apply_period",""),
        # ↓ 디버깅
        # "score":     doc.metadata.get("debug_total_score"),
        # "region":    doc.metadata.get("debug_region_score"),
        # "interest":  doc.metadata.get("debug_interest_score"),
        # "keyword":   doc.metadata.get("debug_keyword_score")
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

# ─────────────────────────────────── #
# 글로벌 임베딩 및 키워드 벡터DB (키워드 전용)
# Load embedding function globally
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")
from openai import OpenAI
client = OpenAI(api_key=api_key) if api_key else OpenAI()
embedding = OpenAIEmbeddings()
# Load keyword vectorstore (ensure it's built with keyword terms only)
keyword_vectordb = Chroma(persist_directory="./kwdb", embedding_function=embedding)
category_vectordb = Chroma(persist_directory="./categorydb", embedding_function=embedding)
# Main policy vectorstore
policy_vectordb = Chroma(persist_directory="./chroma_policies", embedding_function=embedding)
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
        "서울특별시 강동구",
        "서울",
        "서울특별시",
        "서울시"
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
        "제주특별자치도 제주시", "제주특별자치도 서귀포시", "제주도",
        "제주",
        "제주도",
        "제주특별자치도"
    ]
}

REGION_ALIASES = {
    "서울": [
        "강남", "강남역", "강남구", "잠실", "잠실역", "송파", "송파역", "압구정", "압구정로데오",
        "홍대", "홍대입구", "홍대입구역", "합정", "합정역", "상암", "상암동", "상암디엠씨", "여의도", "여의나루",
        "신촌", "이태원", "한남", "한남동", "건대", "건대입구", "성수", "성수동", "왕십리", "왕십리역",
        "마포", "마포구", "종로", "광화문", "서초", "서초구", "노량진", "노량진역"
    ],
    "경기": [
        "성남", "성남시", "분당", "판교", "판교역", "정자동", "수내", "야탑", "서현",
        "용인", "기흥", "수지", "죽전", "광교", "광교역", "수원", "일산", "일산동구",
        "일산서구", "덕이", "주엽", "대화", "탄현", "부천", "부천시", "오정", "상동",
        "중동", "광명", "안양", "평택", "의정부", "하남", "김포", "포천"
    ],
    "인천": ["송도", "송도국제도시", "청라", "청라국제도시", "영종", "구월", "부평", "계양"],
    "부산": ["해운대", "해운대구", "해운대역", "광안리", "광안", "서면", "남포", "남포동", "동래", "센텀"],
    "대구": ["동성로", "수성", "달서", "칠성", "대명"],
    "광주": ["상무", "상무지구", "첨단", "송정"],
    "대전": ["둔산", "둔산동", "유성", "탄방"],
    "울산": ["삼산", "삼산동", "태화", "무거"],
    "세종": ["어진동", "나성동", "호수공원", "세종시청"],
    "강원": ["원주", "춘천", "강릉", "속초", "평창", "홍천"],
    "충북": ["청주", "청원", "충주", "제천", "오창"],
    "충남": ["천안", "아산", "홍성", "서산", "당진", "예산"],
    "전북": ["전주", "전주혁신", "익산", "군산", "완주"],
    "전남": ["순천", "여수", "광양", "목포", "무안"],
    "경북": ["포항", "구미", "경주", "영주", "안동"],
    "경남": ["창원", "김해", "진주", "거제", "통영", "양산"],
    "제주": ["서귀포", "애월", "조천", "한림", "성산", "표선", "제주시"]
}

# ---- 단일 키워드 및 'OO시' 변형 자동 추가 ----
# 각 표준 지역명 자체(예: '서울', '경기')와 흔히 쓰는 '○○시' 변형을 REGION_MAPPING 리스트에 자동 포함시켜
# 단일 키워드 입력도 인식하도록 확장합니다.
for std_region, names in REGION_MAPPING.items():
    # 1) 단일 표준 지역명 추가
    if std_region not in names:
        names.append(std_region)
    # 2) '○○시' 변형 추가 (도 단위는 제외)
    if not std_region.endswith("도") and not std_region.endswith("시"):
        si_variant = f"{std_region}시"
        if si_variant not in names:
            names.append(si_variant)

# 지역 이름 역매핑 (예: '여주시' → '경기')
REVERSE_REGION_LOOKUP = {}
for std_region, full_names in REGION_MAPPING.items():
    for name in full_names:
        tokens = re.findall(r"[가-힣]{2,}", name)
        for token in tokens:
            if token not in REVERSE_REGION_LOOKUP:
                REVERSE_REGION_LOOKUP[token] = std_region

            stripped = re.sub(r"(특별자치도|특별자치시|특별시|광역시)$", "", token)
            if stripped and stripped not in REVERSE_REGION_LOOKUP:
                REVERSE_REGION_LOOKUP[stripped] = std_region

            base_tok = re.sub(r"(시|군|구|동|읍|면|리)$", "", stripped)
            if base_tok and base_tok not in REVERSE_REGION_LOOKUP:
                REVERSE_REGION_LOOKUP[base_tok] = std_region

            city_match = re.match(r"([가-힣]+시)", token)
            if city_match:
                city = city_match.group(1)
                if city not in REVERSE_REGION_LOOKUP:
                    REVERSE_REGION_LOOKUP[city] = std_region
                city_base = re.sub(r"(특별시|광역시|시)$", "", city)
                if city_base and city_base not in REVERSE_REGION_LOOKUP:
                    REVERSE_REGION_LOOKUP[city_base] = std_region

            county_matches = re.findall(r"[가-힣]+군", token)
            for county in county_matches:
                if county not in REVERSE_REGION_LOOKUP:
                    REVERSE_REGION_LOOKUP[county] = std_region
                county_base = re.sub(r"군$", "", county)
                if county_base and county_base not in REVERSE_REGION_LOOKUP:
                    REVERSE_REGION_LOOKUP[county_base] = std_region

            district_matches = re.findall(r"[가-힣]+구", token)
            for district in district_matches:
                if district not in REVERSE_REGION_LOOKUP:
                    REVERSE_REGION_LOOKUP[district] = std_region
                district_base = re.sub(r"구$", "", district)
                if district_base and district_base not in REVERSE_REGION_LOOKUP:
                    REVERSE_REGION_LOOKUP[district_base] = std_region

            town_matches = re.findall(r"[가-힣]+동", token)
            for town in town_matches:
                if town not in REVERSE_REGION_LOOKUP:
                    REVERSE_REGION_LOOKUP[town] = std_region
                town_base = re.sub(r"동$", "", town)
                if town_base and town_base not in REVERSE_REGION_LOOKUP:
                    REVERSE_REGION_LOOKUP[town_base] = std_region

        # 전체 명칭도 직접 매핑
        if name not in REVERSE_REGION_LOOKUP:
            REVERSE_REGION_LOOKUP[name] = std_region

# 추가: 단일 지명 토큰 매핑
REVERSE_REGION_LOOKUP.setdefault("제주", "제주")
REVERSE_REGION_LOOKUP.setdefault("제주도", "제주")
REVERSE_REGION_LOOKUP.setdefault("서울", "서울")
REVERSE_REGION_LOOKUP.setdefault("서울시", "서울")

for std_region, aliases in REGION_ALIASES.items():
    for alias in aliases:
        if alias and alias not in REVERSE_REGION_LOOKUP:
            REVERSE_REGION_LOOKUP[alias] = std_region
        compact = alias.replace(" ", "")
        if compact and compact not in REVERSE_REGION_LOOKUP:
            REVERSE_REGION_LOOKUP[compact] = std_region

# ─────────────────────────────────── #
# 2. 정책 키워드 · 카테고리
# ─────────────────────────────────── #
KEYWORDS = [
    "바우처", "해외진출", "장기미취업청년", "맞춤형상담서비스", "교육지원",
    "출산", "보조금", "중소기업", "벤처", "대출", "금리혜택",
    "인턴", "공공임대주택", "육아", "청년가장", "신용회복"
]

CATEGORIES = ["일자리", "복지문화", "참여권리", "교육", "주거"]

INTEREST_EXPANSION = {
    "운동": ["생활체육", "운동처방", "운동용품 대여", "건강", "체력", "헬스"],
    "창업": ["창업지원", "창업교육", "사업자등록", "스타트업"],
    "취업": ["일자리", "직무교육", "인턴십", "일경험", "청년고용"],
    "주거": ["임대", "청년주택", "보증금지원", "전세", "월세"],
    "복지": ["심리상담", "정신건강", "건강검진", "생활비지원"]
}

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
    """
    정책 JSON의 category 필드(쉼표 구분 텍스트)를 그대로 리스트로 반환.
    사전에 정의된 CATEGORIES에 없더라도 저장해 두고, 필터 단계에서 매칭합니다.
    """
    if not cat_field:
        return []
    return [c.strip() for c in cat_field.split(",") if c.strip()]

# ─────────────────────────────────── #
# 3. 벡터스토어 로드/생성 (강화 버전)
# ─────────────────────────────────── #
def load_or_build_vectorstore(json_path: str,
                              persist_dir: str,
                              api_key: str) -> Chroma:
    os.environ["OPENAI_API_KEY"] = api_key
    embedding = OpenAIEmbeddings()

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embedding)

    with open(json_path, encoding="utf-8") as f:
        policies = json.load(f)

    def safe_int(val, default=0):
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)

    for p in tqdm(policies, desc="Vectorizing policies"):
        text = (
            f"정책명: {p['title']}\n"
            f"정책ID: {p.get('policy_id')}\n"
            f"지원대상: {safe_int(p.get('min_age'))}세~{safe_int(p.get('max_age'), 99)}세 / "
            f"지역 {', '.join(p.get('region_name', []))}\n"
            f"소득 분위: {p.get('income_condition', '제한 없음')}\n"
            f"혜택: {p.get('support_content', '')}\n"
            f"신청방법: {p.get('apply_method', '')}\n"
            f"설명: {p.get('description', '')}\n"
            f"링크: {p.get('apply_url', '')}"
        )
    
        existing_keywords = p.get("keywords", "")
        if isinstance(existing_keywords, str):
            existing_keywords = [kw.strip() for kw in existing_keywords.split(",") if kw.strip()]
        merged_keywords = list(set(existing_keywords + extract_keywords(text)))
        metadata = {
            "policy_id":        p.get("policy_id"),
            "title":            p["title"],
            "region":           ", ".join(p.get("region_name", [])),
            "categories":       ", ".join(extract_categories(p.get('category', ''))),
            "keywords":         ", ".join(merged_keywords),
            "min_age":          safe_int(p.get("min_age")),
            "max_age":          safe_int(p.get("max_age"), 99),
            "income_condition": p.get("income_condition", "제한 없음"),
            "summary": (p.get("support_content") or p.get("description", ""))[:200],
            "apply_period":     p.get("apply_period", ""),
            "apply_url":        p.get("apply_url", ""),
        }

        # Ensure metadata values are primitive types for Chroma
        for mk, mv in metadata.items():
            if isinstance(mv, (list, set)):
                metadata[mk] = ", ".join(map(str, mv))

        chunks = splitter.split_text(text)
        documents = [Document(page_content=chunk, metadata=metadata) for chunk in chunks]
        vectordb.add_documents(documents)

    vectordb.persist()
    return vectordb

# ─────────────────────────────────── #
# 4. 사용자 입력 파싱
# ─────────────────────────────────── #
from typing import Tuple, Optional, List

# 사용자 입력 클린업 함수 추가
import re
def clean_user_input(text: str) -> str:
    # Remove common conversational endings and particles that interfere with matching
    return re.sub(r"(에\s*사는?|야|인데|이야|임|입니다|거든|임다|라구|라고)", "", text)

# 조사 등을 제거하고 핵심 단어(예: '여주에' → '여주') 추출
def normalize_korean_tokens(text: str) -> List[str]:
    """
    조사·행정구역 접미사(시·군·구·동·읍·면·리) 제거 후 핵심 단어 추출
    """
    tokens = re.findall(r"[가-힣]{2,}", text)
    normalized = []
    for tok in tokens:
        # 조사 제거
        core = re.sub(r"(에|에서|에게|로|으로|의|를|을|이|가|은|는|도|만|이나|까지|부터)$", "", tok)
        # 행정구역 접미사 제거
        core = re.sub(r"(시|군|구|동|읍|면|리)$", "", core)
        if core and core not in normalized:
            normalized.append(core)
    return normalized

# 지역 추출 보조 함수
def extract_region(user_input: str, REGION_MAPPING: dict) -> str:
    cleaned_input = clean_text_for_matching(user_input)
    for std_region, keywords in REGION_MAPPING.items():
        for keyword in keywords:
            if keyword in cleaned_input:
                return std_region
    for std_region, aliases in REGION_ALIASES.items():
        for alias in aliases:
            compact = alias.replace(" ", "")
            if alias in cleaned_input or compact in cleaned_input:
                return std_region
    return ""

def parse_user_input(text: str) -> Tuple[Optional[int], Optional[str], Optional[List[str]]]:
    # 텍스트 정규화: 앞부분에서 strip 및 특수문자 제거
    text = text.strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    # 텍스트 전처리: 조사 제거 및 공백 정리
    text = re.sub(r"[^\w가-힣]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    age = None
    # ① '26살', '26 세' 형태
    if m := re.search(r"(?:만\s*)?(\d{1,2})\s*(?:세|살)", text):
        age = int(m.group(1))
    # ② 단일 숫자만 있는 경우도 나이로 간주 (15~39세 범위)
    if age is None:
        m2 = re.search(r"\b(\d{1,2})\b", text)
        if m2:
            age_cand = int(m2.group(1))
            if 15 <= age_cand <= 39:
                age = age_cand

    # 지역 추출 부분 교체: REGION_MAPPING의 모든 시/군/구 이름이 포함되도록 확장되어 있어야 함
    region = ""
    for std_region, keywords in REGION_MAPPING.items():
        if any(keyword in text for keyword in keywords):
            region = std_region
            break
    if not region:
        for std_region, aliases in REGION_ALIASES.items():
            if any(alias in text for alias in aliases):
                region = std_region
                break
    # Fallback: 단일 토큰 기반 지역 추출
    if not region:
        for token in normalize_korean_tokens(text):
            if token in REVERSE_REGION_LOOKUP:
                region = REVERSE_REGION_LOOKUP[token]
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

# ─────────────────────────────────── #
# 🔧 추천 가능한 관심사 리스트 헬퍼
# ─────────────────────────────────── #
def suggest_remaining_interests(current: List[str]) -> str:
    """
    현재 stored_interests 를 기준으로 아직 제안하지 않은
    INTEREST_MAPPING 상위 카테고리를 콤마로 나열해 반환.
    5개까지만 보여주고 나머지는 '등'으로 표기.
    """
    remaining = [k for k in INTEREST_MAPPING.keys() if k not in current]
    shown = remaining[:5]
    suggestion = ", ".join(shown)
    if len(remaining) > 5:
        suggestion += " 등"
    return suggestion


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

[TASK - Chain of Thought 방식]
사용자의 조건(나이, 지역, 관심사)을 바탕으로 아래 순서대로 추론하며 정책을 추천하세요:

1. 먼저 사용자의 나이가 각 정책의 나이 조건(min_age~max_age)에 부합하는지 확인합니다.
2. 다음으로 지역 조건이 일치하는지 확인합니다. 정확한 지역이 없으면 전국 공통 정책을 포함합니다.
3. 관심사 또는 세부 관심사가 정책 키워드 또는 설명에 포함되어 있는지 판단합니다.
4. 위 조건들에 기반해 적합한 정책을 우선순위로 정렬한 후, 상위 3건을 추천합니다.
5. 각 정책은 추천 이유(나이/지역/관심사 조건에 어떻게 부합하는지)를 한 줄로 설명해 주세요.
6. 조건이 명확하지 않으면 조회량이 많은 전국 공통 정책 3건을 대신 추천하세요.

[OUTPUT FORMAT - MARKDOWN]
- 정책명 (소득: ○○): 지원내용 요약 — 추천 이유 (링크 : apply_url) (정첵ID : policy_id)
- 정책명 (소득: ○○): 지원내용 요약 — 추천 이유 (링크 : apply_url) (정첵ID : policy_id)
- 정책명 (소득: ○○): 지원내용 요약 — 추천 이유 (링크 : apply_url) (정첵ID : policy_id)

[EXCEPTION]
- 조건에 맞는 정책이 없을 경우:
    대신 전국 공통 정책 3건을 출력하세요.

[EXAMPLE - NORMAL]
- 청년내일채움공제 (소득: 제한 없음): 중소기업 근무 청년에게 목돈 마련 지원 — 나이와 소득 조건 모두 부합 (출처: policy_123)
- 국민취업지원제도 (소득: 기준중위소득 100% 이하): 취업준비 중 청년에게 맞춤형 취업지원 — 관심사 '취업'과 일치 (출처: policy_456)
- 청년구직활동지원금 (소득: 기준중위소득 120% 이하): 구직활동비 월 최대 50만원 지원 — 지역, 관심사 모두 일치 (출처: policy_789)

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
    retriever = vectordb.as_retriever(search_kwargs={"k": 30})
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
# 가중치: 지역 0.6(전국 포함), 관심사 0.35, 키워드 0.05
MIN_SCORE = 0.3  # 총합 1.0 중 0.3 이상이면 채택

W_REGION   = 0.6
W_INTEREST = 0.35
W_KEYWORD  = 0.05


def jaccard_similarity(a: set, b: set) -> float:
    """두 집합의 자카드 유사도(0~1). 공집합이면 0."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def filter_docs(docs,user_age: Optional[int], user_text: str, region: str, interests: List[str]):
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
        max_age = d.metadata.get("max_age", 999)
        if user_age is not None and user_age > 0:
            if not (min_age <= user_age <= max_age):
                continue

        # ─────────────────────── #
        # 1. 지역 점수 (R: 0 | 0.5 | 1)
        # ─────────────────────── #
        doc_region_str = d.metadata.get("region", "")
        is_nationwide = ("전국" in doc_region_str) or (doc_region_str.strip() == "")
        if is_nationwide:
            region_score = 1.0  # 전국 정책은 동일 가중치
        elif region and any(k in doc_region_str for k in REGION_MAPPING.get(region, [])):
            region_score = 1.0
        elif region and region in doc_region_str:          # 느슨한 포함
            region_score = 0.5
        else:
            region_score = 0.0

        # ─────────────────────── #
        # 2. 관심사 점수 (I: 0~1)
        # ─────────────────────── #
        # 'categories'가 str(list) 형태로 들어올 수도 있어 파싱 진행
        cat_raw = d.metadata.get("categories", [])
        if isinstance(cat_raw, str):
            cat_tokens = [c.strip() for c in re.split(r"[,\[\]'\"\s]+", cat_raw) if c.strip()]
        else:
            cat_tokens = cat_raw
        policy_tags = set(cat_tokens)

        if policy_tags:
            interest_score = jaccard_similarity(interests_set, policy_tags)
        else:
            # 카테고리가 비어 있으면 문서 본문에 관심사 키워드가 직접 포함되어 있는지 계산
            hits = sum(1 for i in interests_set if i in d.page_content)
            interest_score = hits / len(interests_set) if interests_set else 0.0

        # ─────────────────────── #
        # 3. 키워드 점수 (K: 0~1)
        # ─────────────────────── #
        if kw_hits:
            key_raw = d.metadata.get("keywords", [])
            if isinstance(key_raw, str):
                key_tokens = [k.strip() for k in re.split(r"[,\[\]'\"\s]+", key_raw) if k.strip()]
            else:
                key_tokens = key_raw
            doc_keywords = set(key_tokens)
            keyword_score = len(doc_keywords & set(kw_hits)) / len(kw_hits)
        else:
            keyword_score = 0.0

        # ─────────────────────── #
        # 4. 최종 점수 (동적 가중치)
        # ─────────────────────── #
        total_w = 0
        score_sum = 0
        if region:
            total_w += W_REGION
            score_sum += W_REGION * region_score
        if interests_set:
            total_w += W_INTEREST
            score_sum += W_INTEREST * interest_score
        if kw_hits:
            total_w += W_KEYWORD
            score_sum += W_KEYWORD * keyword_score
        # 모든 항목이 비어 있으면 키워드만이라도 사용
        if total_w == 0:
            total_w = W_KEYWORD
            score_sum = W_KEYWORD * keyword_score
        score = score_sum / total_w

        # 디버깅용 점수 메타데이터 저장
        d.metadata["debug_region_score"]   = round(region_score,   3)
        d.metadata["debug_interest_score"] = round(interest_score, 3)
        d.metadata["debug_keyword_score"]  = round(keyword_score,  3)
        d.metadata["debug_total_score"]    = round(score,         3)

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


def console_chat(rag_chain, llm, keyword_vectordb=None, category_vectordb=None, policy_vectordb=None):
    print("\n챗봇이 시작되었습니다. 종료하려면 '종료'를 입력하세요.\n")

    exit_terms = {"종료", "exit", "quit"}
    confirm_terms = {"네", "예", "yes", "y"}
    confirm_all_terms = {"전체", "전부", "all"}
    more_terms = {"더", "more"}

    stored_age = None
    stored_region = None
    stored_interests = []
    stored_income = None
    stored_status = None
    stored_gender = None
    stored_education = None
    stored_desired_job = None
    stored_hope_region = None
    stored_family = None
    stored_special = None

    recommended_ids = set()
    pending_full_docs = []
    pending_total = 0

    vectordb = policy_vectordb if policy_vectordb is not None else globals().get("policy_vectordb")
    if vectordb is None:
        print("Bot: 정책 데이터베이스를 찾지 못했습니다. 벡터스토어를 먼저 빌드해 주세요.")
        return

    OPTIONAL_FIELD_LABELS = {
        "income": "소득 분위",
        "status": "고용 상태",
        "education": "학력·전공",
        "desired_job": "희망 직무",
        "hope_region": "희망 근무 지역",
        "family": "가족·부양 정보",
        "special": "특별 대상 여부",
        "gender": "성별",
    }

    def detect_top_interests(text: str) -> list[str]:
        matches = []
        lowered = text.lower()
        for std_i, kws in INTEREST_MAPPING.items():
            if std_i in text and std_i not in matches:
                matches.append(std_i)
                continue
            for kw in kws:
                if kw in text or kw.lower() in lowered:
                    if std_i not in matches:
                        matches.append(std_i)
                    break
        return matches

    def current_profile() -> dict:
        return {
            "age": stored_age,
            "region": stored_region,
            "interests": stored_interests[:],
            "income": stored_income,
            "status": stored_status,
            "gender": stored_gender,
            "education": stored_education,
            "desired_job": stored_desired_job,
            "hope_region": stored_hope_region,
            "family": stored_family,
            "special": stored_special,
        }

    def apply_user_text(text: str) -> dict:
        nonlocal stored_age, stored_region, stored_interests
        nonlocal stored_income, stored_status, stored_gender
        nonlocal stored_education, stored_desired_job, stored_hope_region
        nonlocal stored_family, stored_special

        info = extract_user_info(text)
        if info["age"]:
            stored_age = info["age"]
        if info["region"]:
            stored_region = info["region"]
        if info["income"]:
            stored_income = info["income"]
        if info["status"]:
            stored_status = info["status"]
        if info["gender"]:
            stored_gender = info["gender"]
        if info["education"]:
            stored_education = info["education"]
        if info["desired_job"]:
            stored_desired_job = info["desired_job"]
        if info["hope_region"]:
            stored_hope_region = info["hope_region"]
        if info["family"]:
            stored_family = info["family"]
        if info["special"]:
            stored_special = info["special"]

        if info["interests"]:
            for interest in info["interests"]:
                if interest and interest not in stored_interests:
                    stored_interests.append(interest)

        for match in detect_top_interests(text):
            if match not in stored_interests:
                stored_interests.append(match)

        return current_profile()

    def print_profile_snapshot(profile: dict):
        interests_text = ", ".join(profile["interests"]) if profile["interests"] else "-"
        essentials = f"[🧠 수집 정보] 나이: {profile['age'] or '-'}, 지역: {profile['region'] or '-'}, 관심사: {interests_text}"
        extras = []
        if profile["status"]:
            extras.append(f"고용 상태: {profile['status']}")
        if profile["income"]:
            extras.append(f"소득: {profile['income']}")
        if profile["education"]:
            extras.append(f"학력·전공: {profile['education']}")
        if profile["desired_job"]:
            extras.append(f"희망 직무: {profile['desired_job']}")
        if profile["hope_region"] and profile["hope_region"] != profile["region"]:
            extras.append(f"희망 근무 지역: {profile['hope_region']}")
        if profile["gender"]:
            extras.append(f"성별: {profile['gender']}")
        if profile["family"]:
            extras.append(f"가족·부양: {profile['family']}")
        if profile["special"]:
            extras.append(f"특별 대상: {profile['special']}")
        if extras:
            print(f"{essentials}\n[ℹ️ 추가 정보] " + ", ".join(extras))
        else:
            print(essentials)

    def missing_optional_fields() -> list[str]:
        profile = current_profile()
        labels = []
        for key, label in OPTIONAL_FIELD_LABELS.items():
            if not profile.get(key):
                labels.append(label)
        return labels

    def prompt_required(missing_labels: list[str]):
        optional = missing_optional_fields()
        lines = ["사용자님을 위한 정책을 안내드리기 위해 몇 가지 정보가 필요해요."]
        lines.append("필수 정보: " + ", ".join(missing_labels))
        if optional:
            lines.append("추가로 " + ", ".join(optional) + " 등을 알려주시면 더 정확한 추천이 가능해요.")
        print("Bot:\\n" + "\\n".join(lines) + "\\n")

    def display_policies(docs: list, limit: int = 10):
        nonlocal recommended_ids
        profile = current_profile()
        if not docs:
            print("Bot:\\n표시할 정책이 없어요. 다른 조건을 알려주세요!\\n")
            return
        print("Bot:\\n맞춤 정책을 안내드릴게요!")
        for idx, doc in enumerate(docs[:limit], 1):
            pid = doc.metadata.get("policy_id", "")
            title = doc.metadata.get("title", "알 수 없는 정책")
            summary = doc.metadata.get("summary") or doc.page_content[:120]
            apply_url = doc.metadata.get("apply_url", "")
            if not apply_url:
                m = re.search(r"https?://[^\s)]+", doc.page_content)
                if not m:
                    m = re.search(r"https?://[^\s)]+", summary or "")
                if m:
                    apply_url = m.group(0)
            reason = _compose_reason(doc, profile)
            print(f"{idx}. {title} (정책ID: {pid})")
            print(f"   요약: {summary.strip()}")
            if apply_url:
                print(f"   신청 링크: {apply_url}")
            if reason:
                print(f"   추천 이유: {reason}")
            if pid:
                recommended_ids.add(pid)
        if len(docs) > limit:
            print(f"\n※ 총 {len(docs)}건 중 상위 {limit}건만 표시했어요. 더 보시려면 '전체'라고 입력해 주세요.")
        print("\n다른 조건이 있다면 계속 말씀해 주세요!")

    REQUIRED_LABELS = [("age", "나이"), ("region", "주거 지역"), ("interests", "관심 분야")]

    while True:
        user_input = input("You: ")
        trimmed = user_input.strip()
        normalized = re.sub(r"[.!?]+$", "", trimmed.lower())

        if pending_full_docs and normalized in confirm_terms:
            display_policies(pending_full_docs)
            pending_full_docs = []
            pending_total = 0
            continue
        if pending_full_docs and normalized in confirm_all_terms:
            display_policies(pending_full_docs, limit=len(pending_full_docs))
            pending_full_docs = []
            pending_total = 0
            continue
        if pending_full_docs and normalized in more_terms:
            display_policies(pending_full_docs, limit=min(len(pending_full_docs), 20))
            pending_full_docs = []
            pending_total = 0
            continue

        if normalized in exit_terms:
            print("Bot: 이용해 주셔서 감사합니다. 안녕히 가세요!")
            break

        force_more_request = is_generic_more_request(user_input)
        if not force_more_request and not is_policy_related_question_llm(user_input):
            print("Bot:\\n저는 대한민국 청년 정책 안내를 도와드리는 챗봇이에요! 정책 관련 질문을 해주세요 😊\\n")
            continue

        apply_user_text(user_input)

        predicted_keywords = None
        embedding_model = None
        if keyword_vectordb:
            embedding_model = OpenAIEmbeddings()
            query_vector = embedding_model.embed_query(user_input)
            docs = keyword_vectordb.similarity_search_by_vector(query_vector, k=3)
            if docs:
                predicted_keywords = [doc.page_content for doc in docs]

        if not predicted_keywords and category_vectordb:
            if embedding_model is None:
                embedding_model = OpenAIEmbeddings()
            query_vector = embedding_model.embed_query(user_input)
            docs = category_vectordb.similarity_search_by_vector(query_vector, k=2)
            if docs:
                predicted_keywords = [doc.page_content for doc in docs]

        if not predicted_keywords:
            from langchain.prompts import PromptTemplate
            prompt = PromptTemplate.from_template("""
            [시스템]
            다음 문장에서 관련 있는 관심사를 추출하세요.
            선택 가능한 항목: 창업, 취업, 금융, 복지, 교육, 공간, 문화예술, 주거, 정신건강, 인턴십

            문장:
            {input}

            결과:
            """)
            response = llm.invoke(prompt.format(input=user_input).to_messages())
            predicted_keywords = [i.strip() for i in response.content.split(",") if i.strip()]

        if predicted_keywords:
            std_interests = []
            for kw in predicted_keywords:
                if is_region_keyword(kw):
                    continue
                if kw in INTEREST_MAPPING and kw not in std_interests:
                    std_interests.append(kw)
                    continue
                for std_i, kws in INTEREST_MAPPING.items():
                    if kw in kws and std_i not in std_interests:
                        std_interests.append(std_i)
                        break
            predicted_keywords = std_interests if std_interests else None

        if predicted_keywords:
            new_topic = any(kw not in stored_interests for kw in predicted_keywords)
            if new_topic:
                stored_interests = [kw for kw in stored_interests if not is_region_keyword(kw)]
                for kw in predicted_keywords:
                    if kw not in stored_interests:
                        stored_interests.append(kw)
            print(f"[🔍 추론된 관심사] → {predicted_keywords}")

        print_profile_snapshot(current_profile())

        missing_labels = [label for key, label in REQUIRED_LABELS if (key == "interests" and not stored_interests) or (key == "age" and stored_age is None) or (key == "region" and not stored_region)]
        if missing_labels and not force_more_request:
            pending_full_docs = []
            pending_total = 0
            prompt_required(missing_labels)
            continue

        pending_full_docs = []
        pending_total = 0

        filters_keywords_only = {"categories": {"$in": stored_interests}} if stored_interests else None

        base_prompt = "추천" if force_more_request else user_input
        search_query = build_query(base_prompt, stored_age, stored_region, stored_interests)
        extras = []
        for extra_value in [stored_status, stored_income, stored_desired_job, stored_education, stored_special]:
            if extra_value:
                extras.append(extra_value)
        if stored_hope_region and stored_hope_region != stored_region:
            extras.append(stored_hope_region)
        if extras:
            search_query = f"{search_query} {' '.join(extras)}"

        try:
            if force_more_request:
                raw_docs = vectordb.similarity_search(search_query, k=50)
            else:
                raw_docs = vectordb.similarity_search(search_query, k=50, filter=filters_keywords_only)
        except Exception:
            raw_docs = vectordb.similarity_search(search_query, k=50)

        filtered_docs = filter_docs(
            raw_docs,
            stored_age,
            search_query,
            stored_region if stored_region else "",
            stored_interests,
        )

        candidate_docs = []
        seen_ids = set()
        for doc in filtered_docs:
            pid = doc.metadata.get("policy_id")
            if not pid or pid in recommended_ids or pid in seen_ids:
                continue
            candidate_docs.append(doc)
            seen_ids.add(pid)

        if not candidate_docs:
            fallback_docs = vectordb.similarity_search("청년 정책 전국 공통", k=10)
            candidate_docs = []
            seen_ids = set()
            for doc in fallback_docs:
                pid = doc.metadata.get("policy_id")
                if not pid or pid in recommended_ids or pid in seen_ids:
                    continue
                candidate_docs.append(doc)
                seen_ids.add(pid)
            if candidate_docs:
                print("Bot:\\n조건에 딱 맞는 정책이 없어 전국 공통 정책을 먼저 살펴봤어요.")
            else:
                print("Bot:\\n조건에 맞는 정책을 찾지 못했어요. 나이, 지역, 관심 분야 외에 소득 분위나 희망 직무 등을 더 알려주시면 도움이 될 것 같아요!\\n")
                continue

        pending_full_docs = candidate_docs
        pending_total = len(candidate_docs)

        preview_count = min(len(candidate_docs), 3)
        missing_optional = missing_optional_fields()
        if pending_total == 1:
            count_msg = "사용자님 조건에 맞는 정책이 1건 검색되었어요."
        else:
            count_msg = f"사용자님 조건에 맞는 정책이 {pending_total}건 검색되었어요."
        confirm_msg = f" '네'라고 입력하시면 상위 {preview_count}건을 바로 안내드릴게요."
        if missing_optional:
            refine_msg = "더 세밀한 추천을 원하시면 " + ", ".join(missing_optional[:4])
            if len(missing_optional) > 4:
                refine_msg += " 등"
            refine_msg += "의 정보를 알려주세요!"
        else:
            refine_msg = "추가로 궁금한 조건이 있다면 이어서 입력해 주세요!"
        print(f"Bot:\\n{count_msg}{confirm_msg}\\n{refine_msg}\\n")


def retrieve_with_fallback(query, age, region, interests, vectordb, k=5):
    filters = []

    meta = {}
    if region:
        meta["region"] = {"$contains": region}
    if age:
        meta["min_age"] = {"$lte": age}
        meta["max_age"] = {"$gte": age}
    if interests:
        meta["categories"] = {"$in": interests}
    filters.append(meta)

    if "region" in meta:
        f2 = meta.copy()
        del f2["region"]
        filters.append(f2)

    if "min_age" in meta and "max_age" in meta:
        f3 = meta.copy()
        del f3["min_age"]
        del f3["max_age"]
        filters.append(f3)

    if "categories" in meta:
        filters.append({"categories": {"$in": interests}})

    filters.append({})  # no filters

    for f in filters:
        try:
            docs = vectordb.similarity_search(query, filter=f, k=k)
            if docs:
                return docs
        except Exception as e:
            continue

    return []


# ─────────────────────────────────── #
# 🔗 FastAPI 연동용 단일 요청 처리 함수
# ─────────────────────────────────── #
from typing import Dict

def _compose_reason(doc: Document, user_info: Dict) -> str:
    """
    간단한 추천 사유 문자열 생성
    """
    reasons = []
    age = user_info.get("age")
    region = user_info.get("region")
    interests = user_info.get("interests", [])

    # 나이
    if age is not None:
        min_age = doc.metadata.get("min_age", 0)
        max_age = doc.metadata.get("max_age", 99)
        if min_age <= age <= max_age:
            reasons.append("나이 조건 부합")

    # 지역
    doc_region = doc.metadata.get("region", "")
    if region:
        if "전국" in doc_region or region in doc_region:
            reasons.append("지역 조건 부합")

    # 관심사
    if interests:
        doc_cats = doc.metadata.get("categories", [])
        if isinstance(doc_cats, str):
            doc_cats = [c.strip() for c in doc_cats.split(",") if c.strip()]
        if set(interests) & set(doc_cats):
            reasons.append("관심사 조건 부합")

    return ", ".join(reasons) if reasons else "일부 조건 부합"

def generate_policy_response(
    user_id: str,
    user_input: str,
    *,
    vectordb: Chroma = policy_vectordb,
    keyword_vectordb: Chroma = keyword_vectordb,
    category_vectordb: Chroma = category_vectordb,
) -> dict:
    """
    FastAPI 서버에서 호출 가능한 단일 질의‑응답 함수.
    - user_input: 사용자의 자연어 질문
    - 반환 형식은 업무 요청서에 명시된 JSON 구조를 따른다.
    """

    # ─────────────────────────────── #
    # 👤 세션 메모리 로드 & 머지
    # ─────────────────────────────── #
    session               = SESSION_STORE[user_id]          # dict with 'user_info', 'recommended_ids'
    prev_info             = session["user_info"] or {}
    prev_recommended_ids  = session["recommended_ids"]

    optional_label_map = {
        "income": "소득 분위",
        "status": "고용 상태",
        "education": "학력·전공",
        "desired_job": "희망 직무",
        "hope_region": "희망 근무 지역",
        "family": "가족·부양 정보",
        "special": "특별 대상 여부",
        "gender": "성별",
    }

    # 1) 새 입력에서 정보 추출
    current_info = extract_user_info(user_input)

    # 2) 이전 정보와 병합 (새 값이 있으면 덮어씀)
    merged_info = prev_info.copy()
    for k, v in current_info.items():
        if v:  # 값이 None/빈 리스트/빈 문자열이 아니면
            merged_info[k] = v

    # 3) 이후 로직은 merged_info 사용
    user_info = merged_info
    age       = user_info.get("age")
    region    = user_info.get("region")
    interests = list(user_info.get("interests", []))  # copy

    # 2) 필수 정보 확인 -----------------------------------------------
    # 👉 누적 정보를 세션에 즉시 저장해 부분 입력도 기억
    session["user_info"] = user_info

    # 누락 항목 식별
    missing = []
    if age is None:
        missing.append("age")
    if region is None:
        missing.append("region")
    if not interests:
        missing.append("interests")

    if missing:
        label_map = {"age": "나이", "region": "지역", "interests": "관심사"}
        missing_kor = [label_map[m] for m in missing]
        prompt_text = f"{', '.join(missing_kor)}를 알려주시면 맞춤형 정책을 추천해드릴게요."
        optional_missing = [label for key, label in optional_label_map.items() if not user_info.get(key)]
        if optional_missing:
            prompt_text += f" 추가로 {', '.join(optional_missing[:4])}"
            if len(optional_missing) > 4:
                prompt_text += " 등"
            prompt_text += " 세부 정보를 알려주시면 더 정밀한 추천이 가능해요."
        return {
            "message": prompt_text,
            "missing_info": missing,
        }

    # 3) (선택) 관심사 보강 -------------------------------------------
    #    벡터 DB를 이용해 추가 관심사를 예측하고, 기존 관심사에 병합
    predicted_interests = []
    embedding = None

    if keyword_vectordb:
        embedding = OpenAIEmbeddings()
        qvec = embedding.embed_query(user_input)
        docs = keyword_vectordb.similarity_search_by_vector(qvec, k=3)
        predicted_interests.extend([d.page_content for d in docs])

    if not predicted_interests and category_vectordb:
        if embedding is None:
            embedding = OpenAIEmbeddings()
        qvec = embedding.embed_query(user_input)
        docs = category_vectordb.similarity_search_by_vector(qvec, k=2)
        predicted_interests.extend([d.page_content for d in docs])

    # INTEREST_MAPPING 기반 표준화
    std_preds = []
    for kw in predicted_interests:
        if kw in INTEREST_MAPPING and kw not in std_preds:
            std_preds.append(kw)
            continue
        for std_i, kws in INTEREST_MAPPING.items():
            if kw in kws and std_i not in std_preds:
                std_preds.append(std_i)
                break

    # 병합
    for p in std_preds:
        if p not in interests:
            interests.append(p)

    # 4) 벡터 검색 + 필터링 -------------------------------------------
    search_query = build_query(user_input, age, region, interests)
    extras = []
    for key in ["status", "income", "desired_job", "education", "special"]:
        value = user_info.get(key)
        if value:
            extras.append(value)
    hope_region = user_info.get("hope_region")
    if hope_region and hope_region != region:
        extras.append(hope_region)
    if extras:
        search_query = f"{search_query} {' '.join(extras)}"
    try:
        raw_docs = vectordb.similarity_search(search_query, k=50)
    except Exception:
        # 검색 오류 시 최소한의 질의로 재시도
        raw_docs = vectordb.similarity_search(user_input, k=50)

    docs = filter_docs(raw_docs, age, search_query, region, interests)

    # 🔎 이전에 추천했던 정책은 제외 + 중복 응답 차단
    seen_ids = set()
    filtered_docs = []
    for d in docs:
        pid = d.metadata.get("policy_id")
        if not pid or pid in prev_recommended_ids or pid in seen_ids:
            continue
        filtered_docs.append(d)
        seen_ids.add(pid)
        if len(filtered_docs) == 3:
            break
    docs = filtered_docs

    # 5) 결과가 없을 때 폴백 ------------------------------------------
    if not docs:
        fallback_docs = vectordb.similarity_search("청년 정책 전국 공통", k=3)
        policies = []
        for d in fallback_docs:
            policies.append({
                "policy_id": d.metadata.get("policy_id", ""),
                "title":     d.metadata.get("title", ""),
                "summary":   d.metadata.get("summary", "") or d.page_content[:120],
            })
        # 👉 세션 업데이트 (fallback도 기록)
        session["recommended_ids"].update([p["policy_id"] for p in policies])
        session["user_info"] = user_info
        return {
            "message": "조건에 맞는 정책을 찾지 못했어요. 전국 공통 정책을 보여드릴게요.",
            "fallback_policies": policies,
            "user_info": user_info,
        }

    # 6) 정상 추천 -----------------------------------------------------
    policies = []
    for d in docs:
        apply_url = d.metadata.get("apply_url", "")
        if not apply_url:
            # 🔎 페이지 본문이나 요약에서 URL 패턴 추출
            m = re.search(r"https?://[^\s)]+", d.page_content)
            if not m:
                m = re.search(r"https?://[^\s)]+", d.metadata.get("summary", ""))
            if m:
                apply_url = m.group(0)
        policies.append({
            "policy_id": d.metadata.get("policy_id", ""),
            "title":     d.metadata.get("title", ""),
            "summary":   d.metadata.get("summary", "") or d.page_content[:120],
            "apply_url": apply_url,
            "reason":    _compose_reason(d, user_info),
        })

    # 👉 세션 업데이트
    session["recommended_ids"].update([p["policy_id"] for p in policies])
    session["user_info"] = user_info

    # 병합
    user_info["interests"] = interests

    return {
        "message": "추천 정책을 안내드립니다.",
        "policies": policies,
        "user_info": user_info,
    }
