from chatbot_v3 import generate_policy_response

# 1st call
print(generate_policy_response("kyle123",
       "저는 26살 서울 거주 청년이에요. 취업 지원 정책이 궁금해요."))

# 2nd call (이어 질문)
print(generate_policy_response("kyle123",
       "다른 정책도 더 있을까요?"))