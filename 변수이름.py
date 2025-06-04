import json

def convert_policy(new_policy: dict) -> dict:
    return {
        "policy_id": new_policy.get("plcyNo", ""),
        "title": new_policy.get("plcyNm", ""),
        "description": new_policy.get("plcyExplnCn", ""),
        "keywords": [new_policy.get("plcyKywdNm")] if new_policy.get("plcyKywdNm") else [],
        "category": new_policy.get("mclsfNm", ""),
        "support_content": new_policy.get("plcySprtCn", ""),
        "min_age": new_policy.get("sprtTrgtMinAge", ""),
        "max_age": new_policy.get("sprtTrgtMaxAge", ""),
        "income_condition": new_policy.get("earnCndSeCd", ""),
        "apply_method": new_policy.get("plcyAplyMthdCn", ""),
        "apply_url": new_policy.get("aplyUrlAddr", ""),
        "apply_period": new_policy.get("aplyYmd", ""),
        "organizer": new_policy.get("sprvsnInstCdNm", ""),
        "executor": new_policy.get("operInstCdNm", ""),
        "region_code": new_policy.get("zipCd", "").split(",") if new_policy.get("zipCd") else [],
        "school_condition": [new_policy.get("schoolCd")] if new_policy.get("schoolCd") else [],
        "job_condition": [new_policy.get("jobCd")] if new_policy.get("jobCd") else [],
        "major_condition": [new_policy.get("plcyMajorCd")] if new_policy.get("plcyMajorCd") else [],
        "region_name": new_policy.get("region_name", []),
        # 새 항목 그대로 유지
        "bscPlanCycl": new_policy.get("bscPlanCycl", ""),
        "bscPlanPlcyWayNo": new_policy.get("bscPlanPlcyWayNo", ""),
        "bscPlanFcsAsmtNo": new_policy.get("bscPlanFcsAsmtNo", ""),
        "bscPlanAsmtNo": new_policy.get("bscPlanAsmtNo", ""),
        "plcyAprvSttsCd": new_policy.get("plcyAprvSttsCd", ""),
        "lclsfNm": new_policy.get("lclsfNm", ""),
        "srngMthdCn": new_policy.get("srngMthdCn", ""),
        "sbmsnDcmntCn": new_policy.get("sbmsnDcmntCn", ""),
        "etcMttrCn": new_policy.get("etcMttrCn", ""),
        "addAplyQlfcCndCn": new_policy.get("addAplyQlfcCndCn", ""),
        "ptcpPrpTrgtCn": new_policy.get("ptcpPrpTrgtCn", ""),
        "bizPrdBgngYmd": new_policy.get("bizPrdBgngYmd", ""),
        "bizPrdEndYmd": new_policy.get("bizPrdEndYmd", "")
    }

# 실행 부분
if __name__ == "__main__":
    with open("final_data.json", encoding="utf-8") as f:
        new_policy_data_list = json.load(f)

    if isinstance(new_policy_data_list, list):
        converted_policies = [convert_policy(p) for p in new_policy_data_list]
    else:
        converted_policies = [convert_policy(new_policy_data_list)]

    with open("final_data1.json", "w", encoding="utf-8") as f:
        json.dump(converted_policies, f, indent=2, ensure_ascii=False)

    print("✅ 변환 완료: final_data1.json 생성됨")