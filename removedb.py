import shutil
import os
import time
import traceback
from dotenv import load_dotenv

# 📦 내부 모듈
from chatbot_v3 import load_or_build_vectorstore

# ─────────────────────────────────────────────── #
# 0. 환경 변수 로드 (OPENAI_API_KEY 등)
# ─────────────────────────────────────────────── #
load_dotenv()

JSON_PATH   = "FINAL_key_cat.json"    # 필요 시 다른 JSON으로 교체
BASE_DIR    = "./chroma_policies"     # 기본 벡터 DB 폴더
OPENAI_KEY  = os.getenv("OPENAI_API_KEY", "")

# ─────────────────────────────────────────────── #
# 1. 재빌드 함수 (readonly 오류 자동 처리)
# ─────────────────────────────────────────────── #
def rebuild_vectorstore(persist_dir: str) -> None:
    """지정 폴더를 삭제한 뒤 벡터스토어를 다시 빌드한다.
    'readonly database' 오류 발생 시 타임스탬프 디렉터리로 자동 재시도."""
    # ① 기존 폴더 제거
    shutil.rmtree(persist_dir, ignore_errors=True)

    try:
        # ② 기본 경로로 빌드
        load_or_build_vectorstore(
            json_path=JSON_PATH,
            persist_dir=persist_dir,
            api_key=OPENAI_KEY,
        )
        print(f"✅  벡터스토어가 {persist_dir} 에 정상 빌드되었습니다!")
    except Exception as e:
        if "readonly database" in str(e).lower():
            # ③ DuckDB 잠금으로 쓰기 실패 → 새 경로로 재시도
            alt_dir = f"{persist_dir}_{int(time.time())}"
            print(f"[!] readonly DB 오류. 새 폴더 {alt_dir} 로 재시도합니다.")
            load_or_build_vectorstore(
                json_path=JSON_PATH,
                persist_dir=alt_dir,
                api_key=OPENAI_KEY,
            )
            print(f"✅  벡터스토어가 {alt_dir} 에 정상 빌드되었습니다!")
        else:
            # ④ 예기치 않은 오류는 스택 추적 출력 후 재발생
            traceback.print_exc()
            raise

# ─────────────────────────────────────────────── #
# 2. 실행 엔트리포인트
# ─────────────────────────────────────────────── #
if __name__ == "__main__":
    rebuild_vectorstore(BASE_DIR)