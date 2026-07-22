# -*- coding: utf-8 -*-
"""
7·8·9회 지방선거 청년 정책 공약 텍스트 마이닝 종합 분석
- 대상 컬럼: '세부 공약'
- 분석 항목:
  1) 전체 단어(명사) 빈도
  2) 연도별(시계열) 단어 빈도 변화
  3) 지역별 / 정당별 단어 빈도
  4) TF-IDF 기반 연도별 특징 단어
  5) 단어 동시출현(co-occurrence) 분석
  6) 2-gram(바이그램) 빈도
  7) 워드클라우드 (전체 + 연도별)
  8) 기초 통계(공약 수, 평균 길이 등)
필요 패키지: pandas, matplotlib, kiwipiepy, wordcloud, scikit-learn
설치: pip install pandas matplotlib kiwipiepy wordcloud scikit-learn
한글 폰트: 나눔고딕 등 시스템에 설치되어 있어야 함
"""

import re
from collections import Counter
from itertools import combinations

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from kiwipiepy import Kiwi
from wordcloud import WordCloud

# ─────────────────────────────────────────────
# 0. 설정
# ─────────────────────────────────────────────
CSV_PATH = "7_8_9회_지방선거지역별_청년_정책_정리_-_박지은.csv"
TEXT_COL = "세부 공약"
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # 환경에 맞게 수정
OUT_DIR = "output"

import os
os.makedirs(OUT_DIR, exist_ok=True)

# matplotlib 한글 설정
font_manager.fontManager.addfont(FONT_PATH)
font_name = font_manager.FontProperties(fname=FONT_PATH).get_name()
matplotlib.rcParams["font.family"] = font_name
matplotlib.rcParams["axes.unicode_minus"] = False

# 불용어: 분석 목적상 의미가 약하거나 모든 문서에 등장하는 단어
STOPWORDS = {
    "청년", "지원", "확대", "추진", "조성", "운영", "설치", "구축", "지급",
    "등", "및", "통해", "위해", "중", "현", "별도", "대상", "활용", "마련",
    "제공", "강화", "도입", "참여", "체계", "프로그램", "프로젝트", "사업",
}
# ※ '청년', '지원' 등은 전 문서 공통어라 기본 제외.
#    포함해서 보고 싶으면 STOPWORDS에서 빼면 됨.

MIN_WORD_LEN = 2  # 1글자 명사 제외

# ─────────────────────────────────────────────
# 1. 데이터 로드 & 전처리
# ─────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=[TEXT_COL]).copy()

def clean_text(t: str) -> str:
    t = re.sub(r"=>.*", " ", t)            # '=>' 뒤 메모(작성자 판단) 제거
    t = re.sub(r"[\(\)\[\]{}'\"“”‘’·⋅/\-–—:,.…※‧]", " ", t)
    t = re.sub(r"\d{4}년?", " ", t)         # 연도 숫자 제거
    t = re.sub(r"\s+", " ", t).strip()
    return t

df["정제텍스트"] = df[TEXT_COL].astype(str).apply(clean_text)
df["년도"] = df["년도"].astype(int)

print("=" * 70)
print("[1] 기초 통계")
print("=" * 70)
print(f"총 공약 수: {len(df)}건")
print(f"\n연도별 공약 수:\n{df['년도'].value_counts().sort_index().to_string()}")
print(f"\n지역별 공약 수:\n{df['지역'].value_counts().to_string()}")
print(f"\n정당별 공약 수:\n{df['정당'].fillna('무소속/미기재').value_counts().to_string()}")
df["글자수"] = df["정제텍스트"].str.len()
print(f"\n공약 텍스트 평균 길이: {df['글자수'].mean():.1f}자 (최소 {df['글자수'].min()} / 최대 {df['글자수'].max()})")

# ─────────────────────────────────────────────
# 2. 형태소 분석 (명사 추출)
# ─────────────────────────────────────────────
kiwi = Kiwi()
# 도메인 고유명사 등록(분리 방지)
for w in ["기본소득", "일자리", "임대주택", "청년수당", "테크노밸리", "공공기관",
          "지역인재", "귀농", "창업보육", "동백전", "정주", "공유대학"]:
    kiwi.add_user_word(w, "NNP")

def extract_nouns(text: str) -> list:
    tokens = kiwi.tokenize(text)
    nouns = [t.form for t in tokens
             if t.tag in ("NNG", "NNP")
             and len(t.form) >= MIN_WORD_LEN
             and t.form not in STOPWORDS]
    return nouns

df["명사"] = df["정제텍스트"].apply(extract_nouns)

# ─────────────────────────────────────────────
# 3. 전체 단어 빈도
# ─────────────────────────────────────────────
all_nouns = [w for lst in df["명사"] for w in lst]
total_freq = Counter(all_nouns)

print("\n" + "=" * 70)
print("[2] 전체 단어 빈도 TOP 30")
print("=" * 70)
for i, (w, c) in enumerate(total_freq.most_common(30), 1):
    print(f"{i:>2}. {w:<10} {c}회")

top20 = total_freq.most_common(20)
plt.figure(figsize=(10, 7))
plt.barh([w for w, _ in reversed(top20)], [c for _, c in reversed(top20)], color="#4C72B0")
plt.title("전체 단어 빈도 TOP 20 (세부 공약 기준)", fontsize=14)
plt.xlabel("빈도")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_전체빈도.png", dpi=150)
plt.close()

# ─────────────────────────────────────────────
# 4. 연도별(시계열) 단어 빈도 변화
# ─────────────────────────────────────────────
year_freq = {}
for y, g in df.groupby("년도"):
    year_freq[y] = Counter([w for lst in g["명사"] for w in lst])

print("\n" + "=" * 70)
print("[3] 연도별 단어 빈도 TOP 10")
print("=" * 70)
for y in sorted(year_freq):
    print(f"\n◆ {y}년")
    for i, (w, c) in enumerate(year_freq[y].most_common(10), 1):
        print(f"  {i:>2}. {w:<10} {c}회")

# 시계열 추이 그래프: 전체 상위 단어들이 연도별로 어떻게 변하는지
years = sorted(year_freq)
track_words = [w for w, _ in total_freq.most_common(10)]
plt.figure(figsize=(11, 6))
for w in track_words:
    # 연도별 공약 수가 다르므로 '공약 1건당 등장률'로 정규화
    n_by_year = df["년도"].value_counts()
    vals = [year_freq[y][w] / n_by_year[y] for y in years]
    plt.plot(years, vals, marker="o", label=w)
plt.xticks(years)
plt.title("주요 단어의 연도별 등장률 변화 (공약 1건당 평균 등장 횟수)", fontsize=13)
plt.ylabel("등장률")
plt.legend(ncol=2, fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_시계열변화.png", dpi=150)
plt.close()

# 연도별 TOP10 비교 막대그래프
fig, axes = plt.subplots(1, len(years), figsize=(5 * len(years), 6), sharex=False)
for ax, y in zip(axes, years):
    top = year_freq[y].most_common(10)
    ax.barh([w for w, _ in reversed(top)], [c for _, c in reversed(top)],
            color={2018: "#4C72B0", 2022: "#DD8452", 2026: "#55A868"}.get(y, "gray"))
    ax.set_title(f"{y}년 TOP 10")
plt.suptitle("연도별 단어 빈도 TOP 10 비교", fontsize=15)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_연도별TOP10.png", dpi=150)
plt.close()

# ─────────────────────────────────────────────
# 5. 지역별 / 정당별 단어 빈도
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[4] 지역별 단어 빈도 TOP 7")
print("=" * 70)
for r, g in df.groupby("지역"):
    freq = Counter([w for lst in g["명사"] for w in lst])
    tops = ", ".join(f"{w}({c})" for w, c in freq.most_common(7))
    print(f"◆ {r} ({len(g)}건): {tops}")

print("\n" + "=" * 70)
print("[5] 정당별 단어 빈도 TOP 10")
print("=" * 70)
for p, g in df.fillna({"정당": "무소속/미기재"}).groupby("정당"):
    freq = Counter([w for lst in g["명사"] for w in lst])
    tops = ", ".join(f"{w}({c})" for w, c in freq.most_common(10))
    print(f"◆ {p} ({len(g)}건): {tops}")

# ─────────────────────────────────────────────
# 6. TF-IDF: 연도별 '특징' 단어 (그 해에만 유독 두드러진 단어)
# ─────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer

    year_docs = {y: " ".join([w for lst in g["명사"] for w in lst])
                 for y, g in df.groupby("년도")}
    vec = TfidfVectorizer(token_pattern=r"\S+")
    m = vec.fit_transform([year_docs[y] for y in years])
    vocab = vec.get_feature_names_out()

    print("\n" + "=" * 70)
    print("[6] TF-IDF 기반 연도별 특징 단어 TOP 10 (해당 연도만의 키워드)")
    print("=" * 70)
    for i, y in enumerate(years):
        row = m[i].toarray().ravel()
        idx = row.argsort()[::-1][:10]
        print(f"◆ {y}년: " + ", ".join(f"{vocab[j]}({row[j]:.3f})" for j in idx))
except ImportError:
    print("\n(scikit-learn 미설치: TF-IDF 분석 생략)")

# ─────────────────────────────────────────────
# 7. 동시출현(co-occurrence) 분석: 같은 공약 안에 함께 나온 단어쌍
# ─────────────────────────────────────────────
pair_counter = Counter()
for lst in df["명사"]:
    for a, b in combinations(sorted(set(lst)), 2):
        pair_counter[(a, b)] += 1

print("\n" + "=" * 70)
print("[7] 단어 동시출현 TOP 20 (같은 공약 내 함께 등장)")
print("=" * 70)
for i, ((a, b), c) in enumerate(pair_counter.most_common(20), 1):
    print(f"{i:>2}. {a} ↔ {b}: {c}회")

# ─────────────────────────────────────────────
# 8. 2-gram(연속 단어쌍) 빈도
# ─────────────────────────────────────────────
bigrams = Counter()
for lst in df["명사"]:
    for i in range(len(lst) - 1):
        bigrams[(lst[i], lst[i + 1])] += 1

print("\n" + "=" * 70)
print("[8] 바이그램(연속 2단어) TOP 15")
print("=" * 70)
for i, ((a, b), c) in enumerate(bigrams.most_common(15), 1):
    print(f"{i:>2}. {a} {b}: {c}회")

# ─────────────────────────────────────────────
# 9. 워드클라우드 (전체 + 연도별)
# ─────────────────────────────────────────────
def make_wc(freq, title, fname):
    if not freq:
        return
    wc = WordCloud(font_path=FONT_PATH, width=900, height=550,
                   background_color="white", colormap="tab10").generate_from_frequencies(freq)
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

make_wc(total_freq, "전체 워드클라우드", f"{OUT_DIR}/04_워드클라우드_전체.png")
for y in years:
    make_wc(year_freq[y], f"{y}년 워드클라우드", f"{OUT_DIR}/05_워드클라우드_{y}.png")

# ─────────────────────────────────────────────
# 10. 결과 저장 (엑셀/CSV)
# ─────────────────────────────────────────────
pd.DataFrame(total_freq.most_common(), columns=["단어", "빈도"]).to_csv(
    f"{OUT_DIR}/단어빈도_전체.csv", index=False, encoding="utf-8-sig")

rows = []
for y in years:
    for w, c in year_freq[y].most_common():
        rows.append({"년도": y, "단어": w, "빈도": c})
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/단어빈도_연도별.csv", index=False, encoding="utf-8-sig")

print(f"\n분석 완료. 그래프·CSV는 '{OUT_DIR}/' 폴더에 저장되었습니다.")
