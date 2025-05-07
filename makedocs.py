from langchain.schema import Document
import json

def convert_json_to_docs(policies):
    docs = []
    for p in policies:
        # 나이 범위
        age_range = f"{p.get('min_age', '정보 없음')}세 ~ {p.get('max_age', '정보 없음')}세"
        # 지역
        region_name = ", ".join(p.get("region_name", ["전국"]))  # 없으면 "전국" 처리

        content = f"""
정책명: {p.get('title', '제목 없음')}
지원내용: {p.get('support_content', '내용 없음')}
대상: 나이 {age_range}
지역: {region_name}
신청 방법: {p.get('apply_method', '방법 없음')}
신청 기간: {p.get('apply_period', '미정')}
설명: {p.get('description', '설명 없음')}
링크: {p.get('apply_url', '없음')}
"""

        docs.append(Document(
            page_content=content.strip(),
            metadata={
                "policy_id": p.get("policy_id"),
                "url": p.get("apply_url")
            }
        ))
    return docs

# JSON 경로
with open("ms_v3_short.json", "r", encoding="utf-8") as f:
    policy_json = json.load(f)

docs = convert_json_to_docs(policy_json)

# 예시 출력
print(docs[20])  # 20번째 정책 출력