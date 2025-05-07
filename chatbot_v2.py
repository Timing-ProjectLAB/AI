#!/usr/bin/env python3
"""
────────────────────────────────────────
전략
1) 정책 정보를 하나의 포맷팅된 문서로 저장
2) 나이·지역·관심사 필터를 적용해 검색 후보 축소
3) context 하나만 넘겨서 LLM이 답변 생성
4) 최초 1회만 임베딩 후 저장, 이후는 로드만
5) gpt-4o-mini 모델 사용
────────────────────────────────────────
"""

import os, re, json
from dotenv import load_dotenv
from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import ConversationalRetrievalChain

# 관심 키워드와 지역명
INTEREST_MAPPING = {
    "창업": ["창업", "창업 지원", "스타트업", "기업 설립"],
    "취업": ["취업", "일자리", "채용", "고용", "취업 지원", "잡페어"],
    "운동": ["운동", "스포츠", "체육", "피트니스", "헬스"],
    "학업": ["학업", "학습", "공부", "교육", "학위", "대학생활"],
    "프로그램": ["프로그램", "워크숍", "세미나", "캠프", "연수"],
    "장학금": ["장학금", "학비 지원", "등록금 지원", "교육비 지원"],
    "해외연수": ["해외연수", "해외 프로그램", "글로벌 연수", "국제 연수", "교환학생"],
    "인턴십": ["인턴십", "현장실습", "인턴", "산학협력", "인턴쉽"]
}

REGION_MAPPING = {
    "서울": ["서울", "서울특별시"],
    "경기": ["경기", "경기도", "수도권"],
    "인천": ["인천", "인천광역시"],
    "부산": ["부산", "부산광역시"],
    "대구": ["대구", "대구광역시"],
    "광주": ["광주", "광주광역시"],
    "대전": ["대전", "대전광역시"],
    "울산": ["울산", "울산광역시"],
    "세종": ["세종", "세종특별자치시"],
    "강원": ["강원", "강원도"],
    "충북": ["충북", "충청북도"],
    "충남": ["충남", "충청남도"],
    "경북": ["경북", "경상북도"],
    "경남": ["경남", "경상남도"],
    "전북": ["전북", "전라북도"],
    "전남": ["전남", "전라남도"],
    "제주": ["제주", "제주특별자치도", "제주도"]
}

# 벡터스토어 로드 또는 생성
def load_or_build_vectorstore(json_path: str, persist_dir: str, api_key: str) -> Chroma:
    os.environ["OPENAI_API_KEY"] = api_key

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=OpenAIEmbeddings())
        print("저장된 Chroma DB 로드 완료")
    else:
        with open(json_path, encoding="utf-8") as f:
            policies = json.load(f)

        docs: List[Document] = []
        for p in policies:
            text = (
                f"정책명: {p['title']}\n"
                f"지원대상: 나이 {p.get('min_age', '0')}세 ~ {p.get('max_age', '99')}세 / 지역 {', '.join(p['region_name'])}\n"
                f"혜택: {p.get('support_content', '')}\n"
                f"신청방법: {p.get('apply_method', '')}\n"
                f"설명: {p.get('description', '')}\n"
                f"링크: {p.get('apply_url', '')}"
            )
            meta = {
                "policy_id": p["policy_id"],
                "region": ", ".join(p["region_name"]),
                "support_content": p.get("support_content", "")
            }
            docs.append(Document(page_content=text, metadata=meta))

        vectordb = Chroma.from_documents(
            docs,
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_dir,
        )
        vectordb.persist()
        print("새로 Chroma DB 생성 및 저장 완료")

    return vectordb

# 유저 입력 파싱
def parse_user_input(text: str) -> Tuple[int, str, List[str]]:
    # 1. 나이 추출
    age = 0
    if m := re.search(r"(\d{2})\s*세", text):
        age = int(m.group(1))

    # 2. 지역 추출
    region = ""
    for std_region, keywords in REGION_MAPPING.items():
        if any(keyword in text for keyword in keywords):
            region = std_region
            break

    # 3. 관심사 추출
    interests = []
    for std_interest, keywords in INTEREST_MAPPING.items():
        if any(keyword in text for keyword in keywords):
            interests.append(std_interest)

    return age, region, interests

# 프롬프트 설정
SYSTEM = SystemMessagePromptTemplate.from_template("""
당신은 대한민국 청년을 위한 맞춤형 정책 안내 챗봇입니다.
사용자의 질문과 주어진 정책 데이터(context)를 바탕으로, 사용자 조건(나이, 거주 지역, 직업, 관심 분야 등)에 부합하는 정책 정보를 정확하고 간결하게 안내하세요.

[지침]
1. 주어진 context 내에서만 정보를 추론하세요.
2. 존재하지 않는 정보는 생성하지 마세요.
3. 핵심만 간단명료하게 답변하세요.
4. 정중하고 친절한 말투를 사용하세요.
5. 조건에 맞는 정책이 없으면: "조건에 맞는 정책이 없습니다."라고 답변하세요.
6. 출처나 내부 식별자(plcyNo 등)는 출력하지 마세요.
7. 조건에 맞는 정책이 여러 개인 경우, 각각 "정책명: 지원내용" 형식으로 나열하세요.

[예시 1]
User: 저는 제주도에 사는 25살 대학생입니다. 취업 관련 정책이 있나요?
Assistant:
- 정책명: 청년 일경험 지원사업
  지원내용: 제주도 내 청년에게 직무 경험을 제공하며, 참여 시 급여를 지원합니다.

- 정책명: 대학생 취업역량 강화 프로그램
  지원내용: 취업 준비생을 위한 면접 교육, 자기소개서 컨설팅 등을 무료로 제공합니다.

[예시 2]
User: 19살 고등학생인데 창업 지원 받을 수 있을까요?
Assistant: 조건에 맞는 정책이 없습니다.

[예시 3]
User: 28세이고 서울에 거주합니다. 주거 지원 받을 수 있나요?
Assistant:
- 정책명: 청년 월세 한시 특별지원
  지원내용: 소득 기준을 충족한 청년에게 월 최대 20만원의 임대료를 최대 12개월까지 지원합니다.
""")

combine_prompt = ChatPromptTemplate.from_messages([
    SYSTEM,
    HumanMessagePromptTemplate.from_template(
        "context:\n{context}\n\n질문: {question}\n\n"
        "문서를 참고하여 한국어로 간결하게 답변하세요."
    ),
])

# RAG 체인 생성
def create_rag_chain(vectordb: Chroma, api_key: str) -> ConversationalRetrievalChain:
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": combine_prompt, "document_variable_name": "context"},
        return_source_documents=True,
        output_key="answer",
    )
    return rag_chain

# 콘솔 채팅
# 콘솔 채팅 (수정)
def console_chat(rag_chain: ConversationalRetrievalChain):
    print("(Ctrl+C 로 종료)\n")
    while True:
        user = input("You: ")
        age, region, interests = parse_user_input(user)

        res = rag_chain.invoke({"question": user})

        # 지역만 필터, 관심사는 LLM 답변에 맡기기
        filtered_docs = []
        for d in res["source_documents"]:
            doc_region = d.metadata.get("region", "")
            region_match = region in doc_region if region else True
            if region_match:
                filtered_docs.append(d)

        if filtered_docs:
            print(f"\nBot:\n{res['answer']}\n")
            for d in filtered_docs[:3]:
                print(f"- {d.metadata['policy_id']}")
        else:
            print("\nBot: 조건에 맞는 정책을 찾지 못했습니다.\n")

        print("─" * 60)

# main
if __name__ == "__main__":
    JSON_PATH = "ms_v3_short.json"
    PERSIST_DIR = "./chroma_policies"
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    vectordb = load_or_build_vectorstore(JSON_PATH, PERSIST_DIR, OPENAI_API_KEY)
    rag = create_rag_chain(vectordb, OPENAI_API_KEY)
    console_chat(rag)