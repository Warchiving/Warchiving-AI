# Warchiving: 2-Stage Wedding Venue Recommendation System
### 추천시스템 기반 예비 부부들을 위한 웨딩 아카이빙 App

> **🏆 DPG AI Challenge 장려상 수상작**
> 불투명한 웨딩 시장의 정보 비대칭을 해소하고, AI 기반 맞춤형 추천으로 소비자 중심의 거래를 돕는 웨딩 아카이빙 플랫폼입니다.

<img src="images/154.png">
<img src="images/background-warchiving.png">

## 🌐 installation(set up)

```bash
pip install -r requirements.txt

uvicorn main:app --reload # fast api 서버 실행
```
---

## 🌐 Service Overview
### 1. Background: 정보 불균형인 웨딩 시장
웨딩 시장은 공급자와 수요자 간의 정보 비대칭이 심각한 대표적인 시장입니다. 과도한 광고성 정보와 불투명한 가격 구조 속에서 예비부부들은 큰 피로감을 느낍니다. 우리는 이러한 문제를 해결하기 위해 **광고성 정보를 배제하고 실제 유저의 날 것 그대로의 리뷰**를 구조화하여, 신뢰할 수 있는 데이터 기반의 지표를 제공하고자 합니다.

### 2. Challenge: 평생 단 한 번, Cold-Start 문제의 정면 돌파
웨딩 도메인의 가장 큰 특징은 유저당 **평생 단 한 번**의 구매가 일어난다는 점입니다. 
- **문제**: 유저의 과거 구매 이력이 전무한 상황에서 발생하는 **Cold-Start** 문제
- **해결**: 특정 유저의 과거 로그에 의존하는 대신, 유저가 현재 입력하는 **실시간 요구사항**을 정밀하게 분석하여 즉각적인 추천을 제공하는 '하이브리드 추천 시스템'을 설계했습니다.
<img src="images/background-background.png">

---

## 🌐 Data Strategy: 수집 및 카테고리 분류
<img src="images/background-1.png">
* **데이터 확보**: 서울/부산권 280개 업체, 총 1,300여 개의 실유저 리뷰(네이버 카페 등 커뮤니티) 크롤링 및 구조화
* **9개 카테고리 정의**: 리뷰 데이터 내 소비자 언어를 분석하여 가장 빈번하게 언급되는 6개 핵심 속성(Aspect) 정의
    * *분류 항목: 식사, 대중교통, 홀 분위기,하객동선, 주차공간, 신부대기실*
* **분류 이유**: 유저마다 베뉴를 선택하는 '우선순위(Individual Preference)'가 다르기 때문에, 비정형 리뷰를 9개 카테고리로 정형화하여 속성별 점수 산출이 가능하도록 설계했습니다.

---

## 🌐 Recommendation Engine Architecture

본 시스템은 대량의 데이터에서 후보군을 확보하고, 문맥을 분석하여 정밀하게 정렬하는 **2-Stage** 구조를 가집니다.
<img src="images/background-2.png">
* **1단계: Hybrid Retrieval (Candidate Generation)**
    * **BM25 (Sparse)**: '채플홀', '분리예식' 등 도메인 고유 명사를 정확하게 캐칭
    * **BGE-M3 (Dense)**: "분위기가 고급스럽다"와 같은 문장의 의미적 유사성 파악
* **2단계: Cross-Encoder (Reranking)**
    * 인출된 후보 리뷰와 유저 쿼리를 Pair로 묶어 Deep Context를 분석하여 순위를 재정렬
* **3단계: Aggregation & Soft Weight**
    * 리뷰 단위의 점수를 업체 단위로 집계하며, 유저의 복합 의도 충족 여부를 가중치로 반영

---

## 🧠 3. Trial & Error: 시행착오와 해결 과정 (Problem-Solution)

### [Problem 01] 데이터 압축 문제와 카테고리별 필터링 도입
* **문제**: 방대한 리뷰 데이터를 뭉텅이(Bulk)로 처리할 경우, 정보가 압축되면서 세부 속성이 유실되거나 검색 정확도가 급격히 떨어지는 현상이 발생했습니다.
* **해결**: 소비자 언어 분석을 통해 추출한 **6개 핵심 카테고리**를 데이터베이스의 개별 컬럼으로 구조화했습니다. Retrival 단계에서 유저 쿼리에 맞는 '카테고리별 필터링'을 선행 적용함으로써, 불필요한 노이즈를 제거하고 검색 대상 데이터의 순도를 높였습니다.

### [Problem 02] 고유명사 매칭과 문맥 이해의 공존 (Hybrid Search & RRF)
* **문제**: Dense Embedding만 사용 시 "버진로드가 길다"와 "버진로드가 짧다" 두 문장을 비슷하게 인식하는 오류가 발생했고, 그렇다고 고유명사(BM25)만 강조하면 문맥적 의도를 놓치는 딜레마가 있었습니다.
* **해결**: 키워드에 강한 **BM25**와 문맥에 강한 **BGE-M3**를 결합한 **Hybrid Search**를 도입했습니다. 특히 각 카테고리별로 인출된 결과들을 **RRF(Reciprocal Rank Fusion)** 알고리즘으로 통합 계산하여, 고유명사의 정확도와 문맥적 의미를 모두 잡은 최적의 후보군을 도출했습니다.

### [Problem 03] Cross-Encoder를 통한 쿼리-리뷰 유사도 고도화
* **문제**: 1차 Retrival 단계에서 인출된 후보군들은 쿼리와의 세밀한 상호작용(Interaction)을 반영하는 데 한계가 있었습니다.
* **해결**: Retrival 이후 **Cross-Encoder**를 배치하여 유저 쿼리와 후보 리뷰 간의 유사도를 다시 산출하여 재순위화(Reranking)하였습니다. 모델 내 **Self-Attention** 메커니즘을 통해 유저 쿼리와 리뷰 문장 사이의 깊은 문맥적 관계를 파악함으로써, 유저의 의도에 가장 부합하는 리뷰가 최상단에 노출되도록 랭킹 품질을 극대화했습니다.

### [Problem 04] 단순 평균의 함정: Soft Weight 설계
* **문제**: 유저가 2개 이상의 카테고리(예: "식사가 맛있고 주차가 편한 곳")를 검색할 때, 단순히 점수를 합산하면 **한 가지 조건만 압도적인 업체**가 상단에 노출되는 현상이 발생했습니다.
    * **예시**: '식사 점수 95점 + 주차 점수 10점(합계 105점)'인 업체가 '식사 60점 + 주차 60점(합계 120점)'인 업체보다 높은 순위를 차지할 수 있으나, 유저는 두 조건을 모두 적당히 만족하는 후자를 더 선호합니다.
* **해결**: 유저 의도 개수($N$) 대비 업체가 실제로 충족한 속성 수($n$)를 반영하여 가중치를 부여하는 **Soft Weight 로직**을 직접 설계하여 적용했습니다.
    $$Total\_Score = \sum (Aspect\_Top\_Scores) \times (1.0 + \frac{n}{N})$$
* **효과**: 특정 조건에만 편중된 업체보다 유저의 다중 요구사항을 고루 충족하는 업체를 상단에 노출함으로써, 실제 추천 결과의 체감 만족도를 크게 개선했습니다.

---

## 🔄 4. User Flow
1. **의도 입력**: 유저가 자유로운 문장으로 선호하는 베뉴 조건 입력.
2. **속성 추출**: AI가 쿼리 내에서 핵심 키워드 및 의도(Aspect) 분류.
3. **단계적 추천**: Hybrid Retrieval → Cross-Encoder Reranking → Soft Weight Aggregation 수행.
4. **결과 제공**: 최종 추천 Tok-20와 랜덤추천 10개의 상품을 함께 추천되도록 하여 유저 쿼리 이외에도 암묵적으로 하도록 함.
   
---

## 🏆 5. Results & Performance
* **지표**: 최종 리랭킹 후 **Recall 90%, nDCG 0.64, Hit Rate 1.0** 달성.
* **성과**: 비정형 리뷰 데이터를 정형화하여 불공정한 웨딩 플랫폼을 추천 시스템으로 해결하고자 하였으며, [**DPG AI Challenge 장려상**](https://aifactory.space/task/6649/overview)을 수상하였습니다.

---

## 🚀 6. Future Work (TODO)
* **사용자 행동 데이터 수집**: 현재는 검색 기반이나, 앱 배포 후 클릭률(CTR), 상세페이지 체류 시간(스크롤), '찜' 등의 시퀀스 데이터를 축적할 예정입니다.
* **모델 고도화**: 축적된 행위 데이터를 바탕으로 단기간에 유저의 잠재적 선호도를 예측하는 CTR 예측 모델로 고도화할 계획입니다.

---

## 🛠️ 7. Tech Stack
- **Backend**: `FastAPI`, `Python`
- **Frontend**: `React Native`
- **Search & AI**: `BGE-M3`, `BM25`, `Cross-Encoder`
- **Data**: `Selenium`, `BeautifulSoup, OpenAI

---

## 👥 8. Team Members

| 이름 | 역할 | 담당 업무 |
| :--- | :--- | :--- |
| **신우림** | **추천 엔진** | Hybrid Retrieval, Reranking, Aggregation 로직 설계 및 구현 |
| **심지영** | **FE / Design** | React Native UI/UX 설계 및 프론트엔드 개발, 디자인 시스템 구축 |
| **정혜주** | **FE / BE** | 프론트엔드 기능 구현 및 FastAPI 연동, 백엔드 API 설계 |
| **이유진** | **BE** | 서버 아키텍처 설계, 데이터베이스 스키마 구축 및 API 개발 |






