#!/usr/bin/env python3
# main.py

import os
from dotenv import load_dotenv
from chatbot_v3 import load_or_build_vectorstore, create_rag_chain, console_chat
import shutil
def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "")
    JSON_PATH   = "ms_v3_keywords_added.json"
    PERSIST_DIR = "./chroma_policies"
    force_rebuild = False #True일 경우 항상 재빌드
    if force_rebuild and os.path.exists(PERSIST_DIR):
        print("[!] 기존 chroma_policies 디렉토리를 삭제합니다.")
        shutil.rmtree(PERSIST_DIR)

    vectordb  = load_or_build_vectorstore(JSON_PATH, PERSIST_DIR, api_key)
    rag_chain, llm = create_rag_chain(vectordb, api_key)
    console_chat(rag_chain, llm)

if __name__ == "__main__":
    main()