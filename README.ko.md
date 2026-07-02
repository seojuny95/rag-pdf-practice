# PDF 문서 기반 RAG 실습

> [English](README.md) · **한국어**

가상의 회사 **구름테크**의 취업규칙 PDF 를 두고, 직원이 규정을 직접 찾지 않아도 궁금한 걸 물으면 답해주는 사내 Q&A 를 만든다.  
프레임워크(LangChain 등) 없이 `pypdf` + `openai` + `chromadb` 만으로, 텍스트가 벡터가 되고 검색되는 과정을 직접 본다.

```
질문 → 임베딩 → 벡터 DB 검색 → 관련 청크 + 질문을 프롬프트로 → LLM 답변
```

## 진행 순서

노트북은 **두 개**이고, 순서대로 하면 된다.

```
1) exercise/rag_pdf.ipynb       — 기본 RAG. 여기서 환경·커널을 만들고 RAG 흐름을 처음부터 본다.
2) exercise/advanced_rag.ipynb  — 고도화 RAG. 1)에서 만든 같은 커널을 그대로 쓰며 개선을 하나씩 누적한다.
```

**반드시 `rag_pdf.ipynb` 를 먼저** 한다 — 환경 세팅과 커널(**RAG PDF Practice**)이 거기서 만들어지고, 고도화 노트북은 그 커널을 그대로 쓴다.

## 기본 RAG - `rag_pdf.ipynb`

블로그 글 **[RAG, 그게 뭔데?](https://seojuny.dev/what-is-rag)** 의 실습이다. 구름테크 취업규칙 PDF 하나를 두고, 프레임워크 없이 `pypdf` + `openai` + `chromadb` 만으로 텍스트가 벡터가 되고 검색돼 답이 나오는 과정을 실습을 통해 확인해 본다.

흐름은 두 단계다.

```
인덱싱  (1~3)  문서에서 텍스트 추출 → 청킹 → 임베딩해서 벡터 DB(Chroma) 저장
답변 생성 (4~5)  질문 임베딩 → 유사도 검색(top-k) → 관련 청크 + 질문을 프롬프트로 → LLM 답변
```

`exercise/rag_pdf.ipynb` 를 열고 위에서부터 셀을 실행하면 된다.
**환경 세팅(파이썬·pipenv·커널 등록·모델 선택)과 값을 바꿔가며 실험하는 법까지 노트북 상단 `0. 환경 세팅` 셀과 `CONFIG` 셀에 모두 안내되어 있다.**

- `OPENAI_API_KEY` 가 있으면 OpenAI, 없으면 로컬 Ollama(무료)로 자동 동작한다.
- 다루는 내용: 텍스트 추출(`pypdf`) · 글자 수 청킹(`CHUNK_SIZE`·`OVERLAP`) · 임베딩·Chroma 저장 · 유사도 검색(top-k, 코사인 유사도 점수 확인) · 답변 생성(`SYSTEM_PROMPT` 로 환각 억제) · **RAG 있을 때 vs 없을 때** 비교 · `CONFIG` 값을 바꿔가며 직접 실험.
- 내 PDF 로 바꾸려면 `data/` 에 파일을 넣고 `CONFIG` 의 `PDF_PATH` 만 변경하면 된다.
- 값을 바꿨을 때: `CHUNK_SIZE`·`OVERLAP` 은 벡터 DB 자체가 달라지므로 **2(청킹) → 3(인덱싱)** 셀을 다시 실행하고, `N_RESULTS`·`SYSTEM_PROMPT`·`TEMPERATURE` 는 질문할 때만 쓰이므로 **질문 셀만** 다시 실행하면 된다.

## 고도화 실습 — `advanced_rag.ipynb`

블로그 글 **[RAG 고도화](https://seojuny.dev/advanced-rag)** 의 실습이다. 가장 단순한 **기본 RAG(STAGE 0)** 에서 시작해, 부족한 부분을 찾고 고도화를 **하나씩 적용**하며 **직전 단계와 답을 비교**한다. 고도화를 직접 구현해 보고 무엇이 얼마나 좋아지는지 직접 확인한다.

```
STAGE 0 기본 → +표보존 → +구조청킹 → +Contextual → +쿼리재작성 → +하이브리드 → +리랭킹 → +출처인용 = 고도화
```

- **먼저 `exercise/rag_pdf.ipynb` 를 완료해야 한다** — 환경 세팅·커널이 거기에 있고, 같은 커널 **RAG PDF Practice** 를 그대로 쓴다.
- 데이터셋은 `rag_pdf` 와 **분리**돼 있다 — `data_advanced/` 의 가상 회사 **노바테크** 사내 문서 **PDF 7개**(취업규칙·복리후생·급여규정·보안정책·인사규정·출장여비규정·사내FAQ). 여러 페이지에 표가 여럿 들어 있고, 규정·안내문·FAQ 등 형태도 다양하다. 표는 `pymupdf4llm` 으로 Markdown 표로 살려 추출한다.
- 다루는 기법: 표보존 추출·구조 청킹·Contextual Retrieval·쿼리 재작성(분해는 직접 구현)·하이브리드 검색·메타데이터 필터·리랭킹·출처 인용·에이전트형 RAG·평가. 마지막엔 같은 파이프라인을 LangChain 구현한 코드를 살펴본다.
- 검증된 라이브러리를 쓴다: `rank-bm25`(BM25)·`kiwipiepy`(한국어 형태소 토큰화)·`sentence-transformers`(리랭킹, `bge-reranker-v2-m3`)·`langchain`(LangChain 마무리 구현)·`ragas`(평가). 노트북 맨 위 **'라이브러리 설치' 셀**이 이들을 **호환되는 버전으로 한 번에** 설치한다 — `langchain` 은 `ragas`·retriever 컴포넌트와 맞물리게 **0.3 으로 고정**한다. RAGAS·LangChain 마무리 모두 노트북에서 바로 실행된다.
- 메모리 벡터 DB(`EphemeralClient`)를 써서 실험환경이 격리된다(디스크에 남지 않는다).

## 파일 구성

```
rag-pdf-practice/
├── exercise/
│   ├── rag_pdf.ipynb          # 1) 기본 RAG 실습 (0.세팅 → CONFIG → 5단계 → 실험) ← 이 README
│   ├── advanced_rag.ipynb     # 2) 고도화 실습 (STAGE 0 기본 → 개선 누적 → 고도화) ← 이 README
│   └── en/                    # 영문 버전 (English)
│       ├── rag_pdf.ipynb
│       └── advanced_rag.ipynb
├── Pipfile                    # 의존성 정의 (pipenv install 이 읽는다)
├── data/
│   ├── sample-취업규칙.pdf     # 구름테크 취업규칙 (rag_pdf 용)
│   └── en/employment-rules.pdf  # 영문
├── data_advanced/             # 노바테크 사내 문서 (advanced_rag 용, 여러 페이지·여러 표·FAQ)
│   ├── 취업규칙.pdf            # 규정(여러 장·조), [별표] 휴가 요약표
│   ├── 복리후생.pdf            # 안내문 + 경조사·복지포인트·검진 표
│   ├── 급여규정.pdf            # 기본급·연차 가산·수당 표 여러 개
│   ├── 보안정책.pdf            # 정보 등급표·장비 코드표 (SEC-VPN-01 등)
│   ├── 인사규정.pdf            # 직급·평가 등급 표
│   ├── 출장여비규정.pdf        # 국내·해외 여비 표
│   ├── 사내FAQ.pdf            # Q&A 형태
│   └── en/                    # 영문 (7개 문서: employment-rules·benefits·pay-policy·security-policy·hr-policy·travel-expense-policy·internal-faq)
└── .env.example
```

> `data/` 와 `data_advanced/` 의 문서는 실제 규정이 아니라 실습용으로 지어낸 가상의 문서다. `en/` 폴더에는 같은 문서의 영문 번역본이 들어 있다.
> `pipenv install` 은 `Pipfile.lock` 에 고정된 버전으로 설치하므로, 누구나 동일한 환경이 된다. 고도화 실습의 무거운 라이브러리(`langchain`·`ragas`·`sentence-transformers` 등)는 `advanced_rag.ipynb` 맨 위 '라이브러리 설치' 셀이 호환 버전으로 설치한다.

---

이 실습 코드와 노트북은 **[Claude](https://claude.ai)**(Anthropic)와 함께 작성했다.
