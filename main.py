#!/usr/bin/env python3
# main.py

import os
from dotenv import load_dotenv
from chatbot_v3 import load_or_build_vectorstore, create_rag_chain, console_chat

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "")
    JSON_PATH   = "ms_v3_short.json"
    PERSIST_DIR = "./chroma_policies"

    vectordb  = load_or_build_vectorstore(JSON_PATH, PERSIST_DIR, api_key)
    rag_chain, llm = create_rag_chain(vectordb, api_key)
    console_chat(rag_chain, llm)

if __name__ == "__main__":
    main()