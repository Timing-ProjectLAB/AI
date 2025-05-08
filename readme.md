# 적응형 필터링 및 키워드·카테고리 에디션 챗봇

대한민국 청년을 위한 정책 추천 챗봇입니다. LangChain, ChromaDB, OpenAI를 활용해 RAG 기반 대화형 검색을 제공합니다.

## 기능

* LangChain과 `gpt-4o` 모델을 이용한 RAG(검색 기반 생성)
* 나이, 지역, 관심사, 키워드 기반 적응형 필터링
* 모듈화된 구조(`chatbot_v3.py` + `main.py`)로 유지보수 용이
* JSON 경로 및 ChromaDB 저장 디렉터리를 지정할 수 있는 CLI 옵션 지원

## 전제 조건

* Python 3.8 이상
* `requirements.txt`에 명시된 패키지 설치
* OpenAI API 키

## 설치

```bash
git clone <your-repo-url>
cd project_root
pip install -r requirements.txt
```

## 설정

프로젝트 루트에 `.env` 파일을 생성하고 다음을 추가하세요.

```bash
OPENAI_API_KEY=your_openai_api_key
```

* 기본 설정

  * JSON\_PATH: `ms_v3_short.json`
  * PERSIST\_DIR: `./chroma_policies`
* CLI 옵션으로 경로 및 디렉터리 변경 가능

## 프로젝트 구조

```
project_root/
├── chatbot.py        # 핵심 로직: 벡터스토어 생성, RAG 체인, 입력 파싱, 필터링
├── main.py           # 엔트리 포인트: 환경 변수 로드, 벡터스토어 빌드, 콘솔 채팅 실행
├── requirements.txt  # 의존성 목록
└── .env              # 환경 변수 (.gitignore 처리)
```

## 사용법

### 기본 실행

```bash
python3 main.py
```

### CLI 옵션 사용

```bash
python3 main.py --json path/to/policies.json --persist ./data_dir
```

* `--json`    : 정책 JSON 파일 경로
* `--persist` : ChromaDB 저장 디렉터리 경로

## 테스트

현재 테스트 코드가 포함되어 있지 않습니다.
필요 시 `pytest`를 도입하고 `tests/` 디렉터리를 생성하세요.

## 라이선스

MIT 라이선스
