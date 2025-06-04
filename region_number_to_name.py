import json

# 1. 정책 리스트 JSON 로드
with open("final_data1.json", encoding="utf-8") as f:
    policies = json.load(f)  # ← 여기가 리스트임!

# 2. 지역 코드 ↔ 지역명 매핑 JSON
with open("region_codes_short.json", encoding="utf-8") as f:
    region_map = json.load(f)

# 3. 모든 정책에 대해 region_codes → region_names 변환
for policy in policies:
    codes = policy.get("region_code", [])
    names = [region_map.get(code, f"[코드 {code} 없음]") for code in codes]
    policy["region_name"] = names

# 4. 결과 저장
with open("final_data1_short.json", "w", encoding="utf-8") as f:
    json.dump(policies, f, ensure_ascii=False, indent=2)

print(f"[✔] 총 {len(policies)}건의 정책에 지역명 추가 완료.")