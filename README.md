서울시 지방선거 청년 공약 텍스트 마이닝 종합 분석 (2010–2026)
Longitudinal NLP & Text Mining Analysis of Youth Electoral Pledges in Seoul Local Elections**

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Kiwipiepy-green?style=flat-square)
![scikit-learn](https://img.shields.io/badge/Machine%20Learning-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)

서울시 25개 자치구 및 서울시 광역단체장을 대상으로 2010년부터 2026년까지(제5회~제9회 지방선거) 제시된 **청년 정책 공약 데이터(195+건)**를 다각도로 분석한 정밀 텍스트 마이닝 프로젝트입니다. 

자연어 처리(NLP), TF-IDF, 코사인 유사도, 정책 유형 자동 분류 알고리즘을 활용하여 **시대별 청년 정책 패러다임의 변화, 정당/진영별(보수 vs 진보) 시각 차이, 공약의 구체성 지표, 재선 구청장의 공약 변천사**를 규명합니다.


주요 분석 모듈 (Analytical Framework)

본 프로젝트는 총 11개의 체계적인 데이터 분석 파이프라인으로 구성되어 있습니다.

| 모듈 | 분석 항목 | 설명 및 사용 기법 |
| :--- | :--- | :--- |
| **[A]** | **기초 통계 & 데이터 품질** | 연도/진영/자치구별 공약 수, 텍스트 길이 분포 산출 |
| **[B]** | **단어 빈도 & 시계열 추이** | 형태소 분석기(`Kiwi`) 기반 명사 추출 및 연도별 키워드 등장률 추적 |
| **[C]** | **TF-IDF 특징어 추출** | 특정 연도/선거 시기에 유독 두드러진 고유 키워드 도출 |
| **[D]** | **정책 유형 자동 분류** | 8대 정책 카테고리(현금지원, 일자리, 주거, 창업 등) 자동 매핑 및 히트맵 생성 |
| **[E]** | **공약 구체성 지표 (Specificity)** | 수치, 기간, 단위 표현 정규식을 통한 공약의 정량화·구체성(0~3점) 평가 |
| **[F]** | **정당 진영 통합 비교** | 보수 vs 진보 진영 간 공약 유형 비중 및 핵심 키워드 차이 검증 |
| **[G]** | **자치구별 편중 분석** | 25개 자치구별 청년 공약 수집량 및 정책 유형 편중도 확인 |
| **[H]** | **동시출현 & N-Gram** | Co-occurrence Matrix 및 Bigram 분석을 통한 연관 정책 키워드 파악 |
| **[I]** | **유사 공약 재등장 탐지** | TF-IDF + Cosine Similarity 기반 선거 간 공약 재탕/반복 유사도 탐지 |
| **[J]** | **재선 구청장 추적** | 2회 이상 당선된 구청장의 시기별 청년 공약 키워드 변화 트래킹 |
| **[K]** | **시각화 (WordCloud 등)** | 전체 및 연도별 워드클라우드, 히트맵, 시계열 추이 차트 자동 생성 |

---

8대 정책 분류 체계 (Policy Taxonomy)

공약 텍스트 내 핵심 키워드를 기반으로 다음과 같이 8대 영역으로 정책 유형을 규칙 기반(Rule-based) 자동 분류합니다.

1. **현금성 지원:** 수당, 기본소득, 월세지원, 교통비, 응시료 지원 등
2. **일자리:** 취업, 채용, 인턴, 구직, 공공근로 등
3. **주거:** 임대주택, 역세권 청년주택, 셰어하우스, 보증금 등
4. **교육·인재양성:** 아카데미, 직업훈련, 계약학과, 캠퍼스 등
5. **창업:** 벤처, 스타트업, 청년몰, 창업보육, 마을기업 등
6. **문화·공간:** 청년공간, 청년센터, 커뮤니티, 동아리, 축제 등
7. **산업·인프라:** 클러스터, IT, 4차 산업, 디지털, 스마트 단지 등
8. **복지·참여:** 청년정치, 참여위원회, 심리상담, 건강, 자치 등

---

Tech Stack & Dependencies

* **Language:** Python 3.9+
* **NLP / Morphological Analysis:** `kiwipiepy` (지능형 한국어 형태소 분석기)
* **Text Mining & ML:** `scikit-learn` (TfidfVectorizer, Cosine Similarity)
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `wordcloud`

---

프로젝트 구조 (Directory Structure)

```text
.
├── 서울시_지방선거_청년_공약_2010_2026.xlsx  # 원본 데이터셋
├── seoul_youth_policy_analysis.py       # 메인 분석 스크립트
├── requirements.txt                      # 필요 라이브러리 목록
├── README.md                             # 프로젝트 설명 문서
└── output_seoul/                         # 분석 결과 저장 폴더 (자동 생성)
    ├── S01_전체빈도.png
    ├── S02_시계열.png
    ├── S03_유형_연도.png
    ├── S04_wc_전체.png
    ├── S05_wc_연도별.png
    ├── 단어빈도_전체.csv
    ├── 공약별_분류결과.csv
    └── 연도별_유형비중.csv
```

---

시작하기 (Getting Started)

 1. 환경 설정 및 필수 패키지 설치

```bash
# Repository 클론
git clone [https://github.com/YOUR-USERNAME/seoul-youth-policy-nlp-2010-2026.git](https://github.com/YOUR-USERNAME/seoul-youth-policy-nlp-2010-2026.git)
cd seoul-youth-policy-nlp-2010-2026

# 필요 패키지 설치
pip install pandas numpy matplotlib kiwipiepy wordcloud scikit-learn openpyxl
```

 2. 한글 폰트 설정 (Linux / Ubuntu 기준)
시각화 그래프 내 한글 깨짐 방지를 위해 나눔글꼴(`NanumGothic`) 설치가 필요합니다.

```bash
sudo apt-get install -y fonts-nanum
```
*(Windows/macOS 사용 시 코드 내 `FONT` 경로를 시스템 한글 폰트 경로에 맞게 수정하세요.)*

### 3. 스크립트 실행

```bash
python seoul_youth_policy_analysis.py
```
실행 완료 후 `output_seoul/` 디렉토리에 시각화 그래프(PNG) 및 정리된 시계열 데이터(CSV)가 시각적으로 출력·저장됩니다.
