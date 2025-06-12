from chatbot_v3 import generate_policy_response

result = generate_policy_response(
    user_id="kyle123",
    user_input="창업 관련 정책 뭐 있어?"
)
print(result)