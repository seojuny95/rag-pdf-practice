# PDF 문서 기반 RAG 실습

블로그 글 **[RAG(Retrieval-augmented generation)](https://seojuny.dev/what-is-rag)** 의 실습 코드다.
가상의 회사 **구름테크**의 취업규칙 PDF 를 두고, 직원이 규정을 직접 찾지 않아도 궁금한 걸 물으면 답해주는 사내 Q&A 를 만든다.

프레임워크(LangChain 등) 없이 `pypdf` + `openai` + `chromadb` 만으로, 텍스트가 벡터가 되고 검색되는 과정을 직접 본다.

```
질문 → 임베딩 → 벡터 DB 검색 → 관련 청크 + 질문을 프롬프트로 → LLM 답변
```

## 시작하기

`exercise/rag_pdf.ipynb` 를 열고 위에서부터 셀을 실행하면 된다.
**환경 세팅(파이썬·pipenv·커널 등록·모델 선택)과 값을 바꿔가며 실험하는 법까지 노트북 맨 위 `0. 환경 세팅` 셀과 `CONFIG` 셀에 모두 적혀 있다.**

- `OPENAI_API_KEY` 가 있으면 OpenAI, 없으면 로컬 Ollama(무료)로 자동 동작한다.
- 내 PDF 로 바꾸려면 `data/` 에 파일을 넣고 `CONFIG` 의 `PDF_PATH` 만 바꾸면 된다.

## 고도화 실습 — `advanced_rag.ipynb`

기본 RAG 를 끝냈으면 **`exercise/advanced_rag.ipynb`** 로 넘어간다. 블로그 글 **[RAG 고도화 및 최적화](https://seojuny.dev/advanced-rag)** 의 실습이다.

**누적식**으로 간다. 가장 단순한 **기본 RAG(STAGE 0)** 에서 시작해, 개선을 **하나씩 적용**하고 **직전 단계와 답을 비교**한다. 그 개선이 무엇을 고치는지 눈으로 확인하고 채택한 뒤, 그 위에 다음 개선을 얹는다. 마지막 STAGE를 다 쌓으면 그게 **고도화 RAG**다.

```
STAGE 0 기본 → +표보존 → +구조청킹 → +Contextual → +쿼리재작성 → +하이브리드 → +리랭킹 → +출처인용 = 고도화
```

- **먼저 `exercise/rag_pdf.ipynb` 를 해야 한다** — 환경 세팅·커널이 거기에 있고, 같은 커널 **RAG PDF Practice** 를 그대로 쓴다.
- 데이터셋은 `rag_pdf` 와 **분리**돼 있다 — `data_advanced/` 의 가상 회사 **노바테크** 사내 문서 **PDF 7개**(취업규칙·복리후생·급여규정·보안정책·인사규정·출장여비규정·사내FAQ). 여러 페이지에 표가 여럿 들어 있고, 규정·안내문·FAQ 등 형태도 다양하다. 표는 `pymupdf4llm` 으로 Markdown 표로 살려 뽑는다.
- 다루는 기법: 표보존 추출·구조 청킹·Contextual Retrieval·쿼리 재작성(분해는 직접 구현)·하이브리드 검색·메타데이터 필터·리랭킹·출처 인용·에이전트형 RAG·평가. 마지막엔 같은 파이프라인을 LangChain 으로도 보여 준다.
- 검증된 라이브러리를 쓴다: `rank-bm25`(BM25)·`kiwipiepy`(한국어 형태소 토큰화)·`sentence-transformers`(리랭킹, `bge-reranker-v2-m3`)·`langchain`(LangChain 마무리 구현)·`ragas`(평가). 노트북 맨 위 **'라이브러리 설치' 셀**이 이들을 **호환되는 버전으로 한 번에** 깐다 — `langchain` 은 `ragas`·retriever 컴포넌트와 맞물리게 **0.3 으로 고정**한다. RAGAS·LangChain 마무리 모두 노트북에서 바로 실행된다.
- 메모리 벡터 DB(`EphemeralClient`)를 써서 실험끼리 격리된다(디스크에 안 남는다).

## 파일 구성

```
rag-pdf-practice/
├── exercise/
│   ├── rag_pdf.ipynb          # 1) 기본 RAG 실습 (0.세팅 → CONFIG → 5단계 → 실험)
│   └── advanced_rag.ipynb     # 2) 고도화 실습 (STAGE 0 기본 → 개선 누적 → 고도화)
├── Pipfile                    # 의존성 정의 (pipenv install 이 읽는다)
├── data/
│   └── sample-취업규칙.pdf     # 구름테크 취업규칙 (rag_pdf 용)
├── data_advanced/             # 노바테크 사내 문서 (advanced_rag 용, 여러 페이지·여러 표·FAQ)
│   ├── 취업규칙.pdf            # 규정(여러 장·조), [별표] 휴가 요약표
│   ├── 복리후생.pdf            # 안내문 + 경조사·복지포인트·검진 표
│   ├── 급여규정.pdf            # 기본급·연차 가산·수당 표 여러 개
│   ├── 보안정책.pdf            # 정보 등급표·장비 코드표 (SEC-VPN-01 등)
│   ├── 인사규정.pdf            # 직급·평가 등급 표
│   ├── 출장여비규정.pdf        # 국내·해외 여비 표
│   └── 사내FAQ.pdf            # Q&A 형태
└── .env.example
```

> `data/` 와 `data_advanced/` 의 문서는 실제 규정이 아니라 실습용으로 지어낸 가상의 문서다.
> `pipenv install` 은 `Pipfile.lock` 에 고정된 버전으로 설치하므로, 누구나 같은 환경이 된다. 고도화 실습의 무거운 라이브러리(`langchain`·`ragas`·`sentence-transformers` 등)는 `advanced_rag.ipynb` 맨 위 '라이브러리 설치' 셀이 호환 버전으로 깐다.

---

이 실습 코드와 노트북은 **[Claude](https://claude.ai)**(Anthropic)와 함께 작성했다.
