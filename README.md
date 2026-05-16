# AI 투자 리서치 에이전트

- 국내 상장기업 8곳에 대한 가격/공시/재무/거시지표/뉴스를 일배치로 수집하고, 해당 종목에 대한 분석 결과를 제공하는 서비스입니다.
- 모든 인프라는 GCP 소액 크레딧 안에서 동작할 수 있도록 만들었습니다. 

## Data

- 본 프로젝트에서는 샘플 8개 기업의 약 3년치 공시/시세/뉴스만 BigQuery에 저장합니다.

샘플 8종목:

| 종목코드 | 이름           |
| -------- | -------------- |
| 005930   | 삼성전자       |
| 000660   | SK하이닉스     |
| 005380   | 현대차         |
| 035420   | 네이버         |
| 035720   | 카카오         |
| 068270   | 셀트리온       |
| 105560   | KB금융         |
| 012450   | 한화에어로스페이스 |

데이터 출처:

| 출처 | 데이터 | 활용 데이터 시점 | 윈도우 |
| --- | --- | --- | --- |
| PyKRX | 일별 시세 (OHLCV) | T-1 | 직전 ~ 60–90영업일 |
| PyKRX | 외국인/기관/개인 순매수 | T-1 | 직전 5거래일 + 연속매수 streak |
| PyKRX | 공매도 잔고 | T-2 | 직전 5거래일 변동률 |
| DART OpenAPI | 공시 이벤트 | 실시간 | 직전 90일 카테고리별 건수 |
| DART OpenAPI | 분기 재무제표 (XBRL) | 분기 발표 시점 | 직전 최신 분기 단건 |
| FRED | 한미 기준금리, 원/달러, VIX | 일간 or 월간 | 최신값 + 1M/3M 변화 |
| 네이버 금융 RSS | 종목 뉴스 | T-30분 | 직전 7일 |

각 신호 문장에는 `as_of` 메타가 강제로 붙어, 카드 상에서 "외국인은 T-1 기준 직전 7거래일 연속 순매수가 체결됨 (2026-05-14)" 형태로 시점을 명시합니다.

## Structure

```mermaid
flowchart LR
    DATA[("외부 데이터<br/>PyKRX · DART · FRED · 뉴스")] --> BQ[("BigQuery<br/>8종목 × 3년치")]
    BQ --> SIG["사실 정리<br/>결정론적 4 + 뉴스 구조화 추출 1"]
    SIG --> AGENT["AI 추론<br/>(Gemini)"]
    AGENT --> CARD["리서치 카드<br/>매수 근거 / 매도 근거"]
    USER(["사용자"]) -->|종목·날짜 선택| WEB["웹 화면"]
    WEB --> AGENT
    CARD --> WEB
```

- **시세 / 수급 / 공시 및 재무 / 거시** 4개 신호는 규칙 기반 룰을 통해 근거를 작성합니다 (LLM 개입 X).
- **뉴스**는 7일치 기사를 임베딩 후 DBSCAN으로 중복을 묶고, 대표 기사 1건을 고정 스키마로 추출합니다. 
- 매수와 매도에 대한 평가를 따로 작성해 한쪽으로 치우치지 않도록 합니다.
- 평가 결과가 생성된 후, 각 인용이 진짜 근거에 부합하는지, '예상된다' 등의 미래 예측 표현이 섞였는지 한 번 더 검증합니다.

## How to Use

### 1. 사전 필요 요소

- Python 3.11+, uv
- [gcloud CLI](https://cloud.google.com/sdk/docs/install)
- GCP 프로젝트 생성 (본 프로젝트는 소액의 비용이 지출되며, BigQuery/Secret Manager API 사용이 필요하므로 결제 계정 연결이 필요합니다.)

### 2. 외부 API 키 발급

| 키 | 발급처 | 
| --- | --- | 
| Gemini API Key | [Google AI Studio](https://aistudio.google.com/apikey) | 
| DART OpenAPI Key | [opendart.fss.or.kr](https://opendart.fss.or.kr/) | 
| FRED API Key | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) | 
| KRX 회원 ID/PW | [data.krx.co.kr](http://data.krx.co.kr/) | 

### 3. GCP 프로젝트 초기 설정

```bash
# 1) 본인 계정 인증
gcloud auth login

# 2) Application Default Credentials 설정
gcloud auth application-default login

# 3) 콘솔에서 프로젝트 생성 후 해당 프로젝트 선택
gcloud config set project <YOUR_GCP_PROJECT_ID>

# 4) 필요 API 활성화
gcloud services enable \
  bigquery.googleapis.com \
  secretmanager.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com
```

### 4. Secret Manager 키 등록

```bash
# Gemini
printf '%s' '<GEMINI_API_KEY>' | gcloud secrets create gemini-api-key --data-file=-

# DART OpenAPI
printf '%s' '<DART_API_KEY>' | gcloud secrets create dart-api-key --data-file=-

# FRED
printf '%s' '<FRED_API_KEY>' | gcloud secrets create fred-api-key --data-file=-

# KRX (계정생성)
printf '%s' '{"id":"<KRX_ID>","pw":"<KRX_PW>"}' | gcloud secrets create krx-credentials --data-file=-
```

- 키 갱신 시:  `gcloud secrets versions add <NAME> --data-file=-` 

### 5. 프로젝트 설정

저장소 루트에 `.env` 파일 생성 후 아래 정보 입력

```bash
# .env 예시
DART_RAG_PROJECT=<YOUR_GCP_PROJECT_ID>
DART_RAG_BQ_DATASET=dart_rag
DART_RAG_BQ_LOCATION=US
DART_RAG_REGION=us-central1
```

### 6. 의존성 설치 & 데이터 적재

```bash
# 의존성 설치
uv sync

# (최초 1회) BigQuery 데이터셋 + 7개 테이블 생성
uv run python -m src.research.ingest.bq

# (최초 1회) 8개 종목 × 약 150일치 시세/수급/공시/뉴스 + 거시지표 적재
#  - 약 5~15분 소요
#  - 재실행 시 동일 구간을 DELETE 후 재로딩
uv run python -m src.research.ingest.data_loader
```

### 7. 로컬 실행

```bash
uv run uvicorn src.serve.app:app --reload --port 8080
```

- 브라우저에서 `http://localhost:8080/` 접속 -> 종목, 기준일 선택 -> '카드 생성' 클릭. 
- 첫 생성은 약 90초, 동일 종목, 기준일에 대한 재요청은 캐시된 데이터로 응답합니다.

### 8. (Optional) Cloud Run 배포

`infra/cloudrun-service.yaml` 을 참고하여 본인 프로젝트용 이미지/서비스 계정으로 수정 후 배포 가능합니다. (Private url 생성)

```bash
# Artifact Registry 저장소 (최초 1회)
gcloud artifacts repositories create cloud-run-source-deploy \
  --repository-format=docker --location=us-central1

# 이미지 빌드 및 배포
gcloud run deploy dart-analysis \
  --source . --region us-central1 \
  --min-instances 0 --max-instances 3 \
  --memory 1Gi --cpu 1
```

- 서비스 계정에 `roles/bigquery.dataEditor`, `roles/bigquery.jobUser`, `roles/secretmanager.secretAccessor` 롤 필요

## For Develop

```bash
# 린트 / 포맷 / 타입체크
uv run ruff check .
uv run black --check .
uv run mypy src

# 테스트
uv run pytest
```

## API

| Method | URL                              | Description                                                       |
| ------ | --------------------------------- | ---------------------------------------------------------- |
| GET    | `/`                               | 웹 화면 (HTML)                                             |
| GET    | `/health`                         | 헬스체크                                                   |
| GET    | `/stocks`                         | 다루는 8종목 목록                                          |
| POST   | `/research/{stock_code}`          | 리서치 카드 생성 (`?force_refresh=true` 로 캐시 무시 재생성) |
| GET    | `/research/{stock_code}/history`  | 과거에 생성된 카드 목록                                    |