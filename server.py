from chatbot_v3 import generate_policy_response

# print(generate_policy_response("kyle123", "안녕하세요"))
# # → missing: age, region, interests

# print(generate_policy_response("kyle123", "나이는 26살이에요"))
# # → missing: region, interests

# print(generate_policy_response("kyle123", "지역은 제주도입니다."))
# # → missing: interests

# print(generate_policy_response("kyle123", "관심사는 주거 정책입니다."))
# # → 실제 정책 3건 반환

# print(generate_policy_response("kyle123", "나에게 맞는 정책 추천해줘"))
# # → 바로 정책 추천 (누락 없음)
print(generate_policy_response("kyle123", "서울에 사는 27세 청년이 받을 수 있는 지원금은?"))