# 📄 1. 텍스트 처리 (Text Processing)

TTS 시스템에서 텍스트 입력을 음성으로 자연스럽게 변환하기 위해, 가장 먼저 수행하는 과정은 텍스트 정규화 및 토크나이제이션입니다.

---

## 📌 1-1. Text Normalization

**정의:**  
사람이 읽을 수 있지만 기계가 바로 발음하기 어려운 입력 텍스트를, **발음 가능한 형식으로 변환**하는 과정입니다.

### ✅ 주요 처리 대상

| 처리 대상         | 설명                                              | 예시 (입력 → 출력)                            |
|------------------|---------------------------------------------------|-----------------------------------------------|
| 숫자              | 숫자를 발음으로 변환                              | `2025` → "이천이십오"                           |
| 기호              | 기호를 문장 형태로 해석                           | `100%` → "백 퍼센트"                           |
| 날짜/시간         | 날짜 및 시간을 말로 변환                          | `7/11` → "칠월 십일일", `14:30` → "두 시 반"   |
| 약어/이니셜       | 축약어를 음소로 풀어서 읽음                      | `AI` → "에이 아이", `TTS` → "티 티 에스"       |
| 단위 및 화폐      | 단위 및 화폐를 발음 단위로 변환                  | `5kg` → "오 킬로그램", `₩3000` → "삼천 원"     |
| 특수문자 구문     | 이메일, URL 등 특수한 패턴의 텍스트 처리         | `jyj@gist.ac.kr` → "제이 와이 제이 앳 지스트 점 에이 씨 점 케이 알" |

---

### 🧪 예시 1: 한국어 입력 텍스트

> `오늘은 7/11이고, 온도는 29°C, 시간은 14:30이다.`

🔽 정규화 결과:  
> `"오늘은 칠월 십일일이고, 온도는 이십구 도, 시간은 두 시 반이다."`

---

### 🧪 예시 2: 영어 입력 텍스트

> `Dr. Kim was born in 1987 and works at AI Lab.`

🔽 정규화 결과:  
> `"Doctor Kim was born in nineteen eighty-seven and works at A I Lab."`

---

## 📌 1-2. Tokenization

**정의:**  
정규화된 텍스트를 모델이 이해할 수 있도록 **자소, 음절, 음소 등의 단위로 분해**하는 과정입니다.

---

### ✅ 주요 분해 방식

| 분해 수준         | 설명                                | 예시 ("학교")                    |
|------------------|-------------------------------------|----------------------------------|
| 자소 (Jamo)       | 글자를 초성/중성/종성으로 분해       | ㅎ + ㅏ / ㄱ + ㅛ                |
| 음절 (Syllable)   | 한글 음절 단위로 분리               | "학", "교"                      |
| 음소 (Phoneme)    | 발음 단위로 분리 (g2p 사용)         | /h a k/ /k j o/                 |
| 단어/문장         | 고수준 단위 (문장 처리)             | "학교", "에 갑니다"             |

---

### 🧪 예시 1: 한국어 문장 (`학교에 갑니다`)

- **자소 단위:**  
  `ㅎ ㅏ ㄱ ㄱ ㅛ ㅇ ㅔ ㄱ ㅏ ㅂ ㄴ ㅣ ㄷ ㅏ`

- **음절 단위:**  
  `학 / 교 / 에 / 갑 / 니 / 다`

- **음소 단위 (g2p):**  
  `/h a k/ /k j o/ /e/ /k a p/ /n i/ /d a/`

> ※ g2p: grapheme-to-phoneme 변환 필요
  ## 🔤 g2p란? (grapheme-to-phoneme)

**g2p**는 **Grapheme-to-Phoneme**의 약자로,  
**글자(grapheme)** → **발음(phoneme)** 으로 변환하는 과정을 의미합니다.

---

### 📌 왜 필요한가?

TTS 시스템은 단순히 문자를 읽는 것이 아니라,  
**해당 문자가 어떻게 발음되는지**를 이해해야 자연스러운 음성을 생성할 수 있습니다.

---

### 📎 예시

| 입력 (문자)    | g2p 결과 (음소)               | 설명                          |
|----------------|-------------------------------|-------------------------------|
| `학교`         | `/h a k/ /k j o/`             | ‘학’은 /hak/, ‘교’는 /kjo/    |
| `AI`           | `/eɪ/ /aɪ/`                   | 알파벳을 실제 발음으로 변환   |
| `1987`         | `naɪnˈtiːn ˈeɪti ˈsɛvən`       | 숫자를 읽는 방식으로 변환     |

---

### 🧠 핵심 요약

> g2p는 텍스트를 음성으로 바꾸기 위한 **중간 단계**로,  
> TTS 모델이 “무엇을 말해야 할지” 정확히 알 수 있게 도와주는 전처리 과정입니다.

---

## 🧰 g2p 도구 예시

### 🔹 한국어

- `g2pk`  
- `KoG2P`  
- `open-korean-text`  

```python
from g2pk import G2p
g2p = G2p()
print(g2p("학교에 갑니다"))  
# 출력: '학꾜에 감니다'
```

---

### 🔸 영어

- `g2p-en` (CMU dictionary 기반)
- `phonemizer`

```python
from g2p_en import G2p
g2p = G2p()
print(g2p("AI model"))  
# 출력: ['EY1', 'AY1', 'M', 'AA1', 'D', 'AH0', 'L']
```

---

## 🧭 전체 처리 흐름

```
Raw Text
   ↓
[Text Normalization]
   ↓
Normalized Text
   ↓
[g2p: Grapheme-to-Phoneme]
   ↓
Phoneme Sequence
   ↓
Acoustic Model → Vocoder
```


---

### 🧪 예시 2: 영어 문장 (`This is an AI model.`)

- **단어 토큰화:**  
  `["This", "is", "an", "AI", "model", "."]`

- **음소 (g2p 적용):**  
  `["DH IH S", "IH Z", "AE N", "EY AY", "M AA D AH L"]`

---

## 🧭 처리 파이프라인 요약

```
Raw Text
   ↓
[Text Normalization]
   ↓
Normalized Text
   ↓
[Tokenization (자소/음절/음소)]
   ↓
Phoneme / Grapheme Sequence
   ↓
Acoustic Model → Vocoder
```

---

## 🧰 실습 팁 (도구 모음)

- **한국어 정규화 + g2p:**
  - `g2pk`, `KoG2P`, `pykospacing`, `soynlp`, `jamo`
- **영어 정규화 + g2p:**
  - `g2p-en`, `phonemizer`, `NVIDIA NeMo g2p`

---

## 🧪 Python 예제 (g2pk + jamo)

```python
# pip install g2pk jamo

from g2pk import G2p
from jamo import h2j

text = "오늘은 7/11이고, 온도는 29°C, 시간은 14:30이다."

g2p = G2p()
normalized = g2p(text)
print("정규화:", normalized)

syllables = list(normalized.replace(" ", ""))
print("음절:", syllables)

jamos = list(h2j(normalized.replace(" ", "")))
print("자소:", jamos)
```

---

필요시 영어 버전, 음성 시각화, 실시간 처리 예제도 제공 가능합니다.
