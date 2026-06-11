# PDF 문서 기반 RAG 실습

블로그 글 **[RAG(Retrieval-augmented generation)](https://seojuny.dev/what-is-rag)** 의 실습 코드다.
가상의 회사 **구름테크**의 취업규칙 PDF 를 두고, 직원이 규정을 직접 찾지 않아도 궁금한 걸 물으면 답해주는 사내 Q&A 를 만든다.

프레임워크(LangChain 등) 없이 `pypdf` + `openai` + `chromadb` 만으로, 텍스트가 벡터가 되고 검색되는 과정을 직접 본다.

```
질문 → 임베딩 → 벡터 DB 검색 → 관련 청크 + 질문을 프롬프트로 → LLM 답변
```

## 시작하기

`rag_pdf.ipynb` 를 열고 위에서부터 셀을 실행하면 된다.
**환경 세팅(파이썬·pipenv·커널 등록·모델 선택)과 값을 바꿔가며 실험하는 법까지 노트북 맨 위 `0. 환경 세팅` 셀과 `CONFIG` 셀에 모두 적혀 있다.**

- `OPENAI_API_KEY` 가 있으면 OpenAI, 없으면 로컬 Ollama(무료)로 자동 동작한다.
- 내 PDF 로 바꾸려면 `data/` 에 파일을 넣고 `CONFIG` 의 `PDF_PATH` 만 바꾸면 된다.

## 파일 구성

```
rag-pdf-practice/
├── rag_pdf.ipynb              # 실습 본체 (0.세팅 → CONFIG → 5단계 → 실험)
├── Pipfile                    # 의존성 정의 (pipenv install 이 읽는다)
├── data/
│   └── sample-취업규칙.pdf     # 가상 회사 구름테크의 취업규칙 (실습용 예시)
└── .env.example
```

> `data/sample-취업규칙.pdf` 는 실제 규정이 아니라 실습용으로 지어낸 가상의 문서다.
> `pipenv install` 은 `Pipfile.lock` 에 고정된 버전으로 설치하므로, 누구나 같은 환경이 된다.
