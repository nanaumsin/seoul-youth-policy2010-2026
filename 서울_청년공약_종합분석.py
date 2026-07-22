# -*- coding: utf-8 -*-
"""
서울시 지방선거 청년 공약 (2010~2026) — 종합 분석
데이터: 195건, 서울 25개 자치구 + 서울시, 5개 선거(2010/2014/2018/2022/2026)

분석 구성
 [A] 기초 통계 & 데이터 품질
 [B] 단어 빈도 (전체 / 연도별 시계열 / 정당진영별)
 [C] TF-IDF 연도별 특징어
 [D] 정책 유형 자동 분류 → 연도·진영 히트맵
 [E] 공약 구체성 지표 시계열
 [F] 정당 진영 통합(보수/진보) 비교
 [G] 자치구별 공약량 & 유형 편중
 [H] 동시출현 / 바이그램
 [I] 유사도 기반 재등장·반복 공약 탐지
 [J] 재선 구청장 공약 변화 추적
 [K] 워드클라우드 (전체 + 연도별)
결과: output_seoul/ 에 그래프·CSV 저장
"""
import os, re
from collections import Counter
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from kiwipiepy import Kiwi
from wordcloud import WordCloud

XLSX = "서울시_지방선거_청년_공약_2010_2026.xlsx"
FONT = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
OUT = "output_seoul"
os.makedirs(OUT, exist_ok=True)

font_manager.fontManager.addfont(FONT)
matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
matplotlib.rcParams["axes.unicode_minus"] = False

STOPWORDS = {"청년", "지원", "확대", "추진", "조성", "운영", "설치", "구축", "지급",
    "등", "및", "통해", "위해", "중", "현", "별도", "대상", "활용", "마련", "제공",
    "강화", "도입", "참여", "체계", "프로그램", "프로젝트", "사업", "지역", "정책", "구민"}

# 진보/보수 진영 매핑 (시기별 당명 변화 통합)
PROGRESSIVE = {"더불어민주당", "민주당", "새정치민주연합", "열린우리당"}
CONSERVATIVE = {"국민의힘", "새누리당", "한나라당", "미래통합당", "자유한국당"}
def camp(party):
    if party in PROGRESSIVE: return "진보"
    if party in CONSERVATIVE: return "보수"
    return "기타"

# ── 로드 & 전처리 ──────────────────────────────
df = pd.read_excel(XLSX)
df = df.dropna(subset=["세부 공약"]).copy()
df["년도"] = df["년도"].astype(int)
df["진영"] = df["정당"].apply(camp)
# 자치구 정규화: '서울특별시 종로구'/'종로구' → '종로구', '서울특별시'는 광역
def norm_gu(x):
    x = str(x).replace("서울특별시", "").strip()
    return x if x else "서울시(광역)"
df["자치구"] = df["지역"].apply(norm_gu)

def clean(t):
    t = re.sub(r"=>.*", " ", str(t))
    t = re.sub(r"[\(\)\[\]{}'\"“”‘’·⋅/\-–—:,.…※‧]", " ", t)
    t = re.sub(r"\d{4}년?", " ", t)
    return re.sub(r"\s+", " ", t).strip()
df["텍스트"] = df["세부 공약"].apply(clean)

# ── [A] 기초 통계 ──────────────────────────────
print("=" * 72)
print("[A] 기초 통계 & 데이터 품질")
print("=" * 72)
print(f"총 공약: {len(df)}건 | 자치구: {df['자치구'].nunique()}개 | 당선인: {df['당선인 이름'].nunique()}명")
print("\n연도별 공약 수:")
print(df["년도"].value_counts().sort_index().to_string())
print("\n진영별 공약 수:")
print(df["진영"].value_counts().to_string())
print("\n연도 × 진영 교차:")
print(pd.crosstab(df["년도"], df["진영"]).to_string())
df["글자수"] = df["텍스트"].str.len()
print(f"\n공약 평균 길이: {df['글자수'].mean():.1f}자")

# ── 형태소 분석 ────────────────────────────────
kiwi = Kiwi()
for w in ["기본소득","일자리","임대주택","청년수당","공공기관","지역인재","귀농",
          "창업보육","월세지원","역세권","청년몰","셰어하우스","마을기업","사회적기업","반값등록금"]:
    kiwi.add_user_word(w, "NNP")
def nouns(t):
    return [x.form for x in kiwi.tokenize(t)
            if x.tag in ("NNG","NNP") and len(x.form) >= 2 and x.form not in STOPWORDS]
df["명사"] = df["텍스트"].apply(nouns)

# ── [B] 단어 빈도 ──────────────────────────────
allw = [w for l in df["명사"] for w in l]
total = Counter(allw)
print("\n" + "=" * 72); print("[B] 전체 단어 빈도 TOP 30"); print("=" * 72)
for i,(w,c) in enumerate(total.most_common(30),1):
    print(f"{i:>2}. {w:<10}{c}", end="   " if i%3 else "\n")
print()

years = sorted(df["년도"].unique())
yfreq = {y: Counter(w for l in g["명사"] for w in l) for y,g in df.groupby("년도")}
print("\n연도별 TOP 8:")
for y in years:
    print(f"  {y}: " + ", ".join(f"{w}({c})" for w,c in yfreq[y].most_common(8)))

# 전체빈도 그래프
top25 = total.most_common(25)
plt.figure(figsize=(10,8))
plt.barh([w for w,_ in reversed(top25)], [c for _,c in reversed(top25)], color="#4C72B0")
plt.title("서울시 청년공약 단어 빈도 TOP 25 (2010–2026)"); plt.xlabel("빈도")
plt.tight_layout(); plt.savefig(f"{OUT}/S01_전체빈도.png", dpi=150); plt.close()

# 시계열: 주요어 등장률
nby = df["년도"].value_counts()
track = [w for w,_ in total.most_common(12)]
plt.figure(figsize=(12,6))
for w in track:
    plt.plot(years, [yfreq[y][w]/nby[y] for y in years], marker="o", label=w)
plt.xticks(years); plt.title("주요 단어 연도별 등장률 (공약 1건당)"); plt.ylabel("등장률")
plt.legend(ncol=3, fontsize=8); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/S02_시계열.png", dpi=150); plt.close()

# ── [C] TF-IDF 연도별 특징어 ───────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
ydoc = {y: " ".join(w for l in g["명사"] for w in l) for y,g in df.groupby("년도")}
vec = TfidfVectorizer(token_pattern=r"\S+")
M = vec.fit_transform([ydoc[y] for y in years]); vocab = vec.get_feature_names_out()
print("\n" + "=" * 72); print("[C] TF-IDF 연도별 특징어 (그 해에 유독 두드러진 단어)"); print("=" * 72)
for i,y in enumerate(years):
    row = M[i].toarray().ravel(); idx = row.argsort()[::-1][:10]
    print(f"  {y}: " + ", ".join(f"{vocab[j]}" for j in idx))

# ── [D] 정책 유형 분류 ─────────────────────────
CATS = {
 "현금성 지원": ["수당","기본소득","소득","연금","응시료","활동비","자산형성","반값","등록금","월세지원","교통비","면접"],
 "일자리": ["일자리","고용","취업","채용","인턴","공공근로","아르바이트","직업","구직","취업박람회","할당채용"],
 "주거": ["임대주택","임대","주거","월세","역세권","기숙사","셰어하우스","전세","보증금","레지던스"],
 "교육·인재양성": ["대학","인재","양성","아카데미","교육","계약학과","연수","캠퍼스","직업훈련","도서관"],
 "창업": ["창업","벤처","창업보육","스타트업","청년몰","마을기업","사회적기업"],
 "문화·공간": ["문화","공간","센터","광장","커뮤니티","활동","동아리","축제","카페","허브","라운지"],
 "산업·인프라": ["산업","단지","인프라","클러스터","IT","벤처단지","혁신","4차","디지털","스마트"],
 "복지·참여": ["복지","참여","자치","위원회","네트워크","심리","건강","상담","권리"],
}
def classify(t):
    r = [c for c,ks in CATS.items() if any(k in t for k in ks)]
    return r or ["기타"]
df["유형"] = df["세부 공약"].astype(str).apply(classify)
long = pd.DataFrame([{"년도":r["년도"],"진영":r["진영"],"자치구":r["자치구"],"유형":c}
                    for _,r in df.iterrows() for c in r["유형"]])
print("\n" + "=" * 72); print("[D] 정책 유형 분류"); print("=" * 72)
print("\n유형별 전체:")
print(long["유형"].value_counts().to_string())
pv_y = long.pivot_table(index="유형", columns="년도", aggfunc="size", fill_value=0)
pv_y_pct = (pv_y/df.groupby("년도").size()*100).round(1)
print("\n연도 × 유형 (연도내 %):"); print(pv_y_pct.to_string())

fig,ax = plt.subplots(figsize=(9,6))
im = ax.imshow(pv_y_pct.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(pv_y_pct.columns)), pv_y_pct.columns)
ax.set_yticks(range(len(pv_y_pct.index)), pv_y_pct.index)
for i in range(pv_y_pct.shape[0]):
    for j in range(pv_y_pct.shape[1]):
        ax.text(j,i,pv_y_pct.values[i,j],ha="center",va="center",fontsize=9)
ax.set_title("연도 × 정책유형 비중(%)"); plt.colorbar(im, fraction=0.046)
plt.tight_layout(); plt.savefig(f"{OUT}/S03_유형_연도.png", dpi=150); plt.close()

# ── [E] 구체성 지표 ────────────────────────────
def spec(t):
    s=0
    if re.search(r"\d+\s*(만\s*원|억|%)", t): s+=1
    if re.search(r"\d+\s*(년|개월)", t) or re.search(r"20\d\d", t): s+=1
    if re.search(r"\d+\s*(개|명|호|곳|가구|%p?)", t): s+=1
    return s
df["구체성"] = df["세부 공약"].astype(str).apply(spec)
print("\n" + "=" * 72); print("[E] 공약 구체성 (0~3점)"); print("=" * 72)
print(f"전체 평균: {df['구체성'].mean():.2f} | 추상적(0점) 비율: {(df['구체성']==0).mean()*100:.0f}%")
print("\n연도별 평균 구체성:")
print(df.groupby("년도")["구체성"].mean().round(2).to_string())

# ── [F] 진영 비교 ──────────────────────────────
print("\n" + "=" * 72); print("[F] 진영(보수/진보) 비교"); print("=" * 72)
cl = long[long["진영"].isin(["진보","보수"])]
pv_c = cl.pivot_table(index="유형", columns="진영", aggfunc="size", fill_value=0)
nc = df[df["진영"].isin(["진보","보수"])]["진영"].value_counts()
pv_c_pct = (pv_c/nc*100).round(1)
print("유형별 비중(%):"); print(pv_c_pct.to_string())
print("\n진영별 평균 구체성:")
print(df[df["진영"].isin(["진보","보수"])].groupby("진영")["구체성"].mean().round(2).to_string())
print("\n진영별 특징어(TF-IDF):")
cdoc = {c:" ".join(w for l in g["명사"] for w in l) for c,g in df[df["진영"].isin(["진보","보수"])].groupby("진영")}
cv = TfidfVectorizer(token_pattern=r"\S+"); CM = cv.fit_transform(list(cdoc.values())); cvoc = cv.get_feature_names_out()
for i,c in enumerate(cdoc):
    row=CM[i].toarray().ravel(); idx=row.argsort()[::-1][:12]
    print(f"  {c}: " + ", ".join(cvoc[j] for j in idx))

# ── [G] 자치구별 ───────────────────────────────
print("\n" + "=" * 72); print("[G] 자치구별 공약량 TOP 10"); print("=" * 72)
print(df["자치구"].value_counts().head(10).to_string())

# ── [H] 동시출현 / 바이그램 ────────────────────
pair = Counter()
for l in df["명사"]:
    for a,b in combinations(sorted(set(l)),2): pair[(a,b)]+=1
print("\n" + "=" * 72); print("[H] 동시출현 단어쌍 TOP 15"); print("=" * 72)
for i,((a,b),c) in enumerate(pair.most_common(15),1): print(f"  {a} ↔ {b}: {c}")
bi = Counter()
for l in df["명사"]:
    for i in range(len(l)-1): bi[(l[i],l[i+1])]+=1
print("\n바이그램 TOP 12:")
for (a,b),c in bi.most_common(12): print(f"  {a} {b}: {c}")

# ── [I] 유사도 재등장 공약 ─────────────────────
from sklearn.metrics.pairwise import cosine_similarity
df["명사문"] = df["명사"].apply(lambda l: " ".join(l))
mask = df["명사문"].str.strip().astype(bool)
sub = df[mask].reset_index(drop=True)
Xs = TfidfVectorizer(token_pattern=r"\S+").fit_transform(sub["명사문"])
sim = cosine_similarity(Xs); np.fill_diagonal(sim,0)
print("\n" + "=" * 72); print("[I] 유사 공약쌍 TOP 10 (★=연도 교차)"); print("=" * 72)
seen=set(); cnt=0
order = np.dstack(np.unravel_index(np.argsort(sim.ravel())[::-1], sim.shape))[0]
for i,j in order:
    if i>=j: continue
    key=(min(i,j),max(i,j))
    if key in seen: continue
    seen.add(key)
    a,b=sub.iloc[i],sub.iloc[j]
    cross=" ★" if a["년도"]!=b["년도"] else ""
    print(f"  {sim[i,j]:.2f}{cross} [{a['년도']} {a['당선인 이름']}] {a['텍스트'][:32]} ↔ [{b['년도']} {b['당선인 이름']}] {b['텍스트'][:32]}")
    cnt+=1
    if cnt>=10: break

# ── [J] 재선 구청장 ────────────────────────────
print("\n" + "=" * 72); print("[J] 재선 구청장 공약 변화 (2회↑ 당선)"); print("=" * 72)
rep = df.groupby("당선인 이름")["년도"].nunique()
rep = rep[rep>1].index.tolist()
print(f"재등장 당선인 {len(rep)}명: {', '.join(rep)}")
for name in rep[:5]:
    s = df[df["당선인 이름"]==name]
    print(f"\n◆ {name} ({s['자치구'].iloc[0]}, {sorted(s['년도'].unique())})")
    for y,g in s.groupby("년도"):
        wc=Counter(w for x in g["명사"] for w in x)
        print(f"   {y}: " + ", ".join(f"{w}" for w,_ in wc.most_common(6)))

# ── [K] 워드클라우드 ───────────────────────────
def wc(freq,title,f):
    if not freq: return
    w=WordCloud(font_path=FONT,width=900,height=520,background_color="white",colormap="tab10").generate_from_frequencies(freq)
    plt.figure(figsize=(10,6)); plt.imshow(w); plt.axis("off"); plt.title(title,fontsize=14)
    plt.tight_layout(); plt.savefig(f,dpi=150); plt.close()
wc(total,"서울시 청년공약 전체 (2010–2026)",f"{OUT}/S04_wc_전체.png")
for y in years: wc(yfreq[y],f"{y}년",f"{OUT}/S05_wc_{y}.png")

# ── 저장 ───────────────────────────────────────
pd.DataFrame(total.most_common(),columns=["단어","빈도"]).to_csv(f"{OUT}/단어빈도_전체.csv",index=False,encoding="utf-8-sig")
df[["순번","년도","자치구","당선인 이름","정당","진영","세부 공약","유형","구체성"]].to_csv(
    f"{OUT}/공약별_분류결과.csv",index=False,encoding="utf-8-sig")
pv_y_pct.to_csv(f"{OUT}/연도별_유형비중.csv",encoding="utf-8-sig")
print(f"\n완료. → {OUT}/")
