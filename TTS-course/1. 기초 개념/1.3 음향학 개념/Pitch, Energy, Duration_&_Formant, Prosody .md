## 🎙️ 음향학 개념 (Pitch, Energy, Duration, Formant, Prosody)

TTS 음향 모델은 음성의 "내용" 뿐만 아니라, "표현 방식"까지 자연스럽게 생성해야 합니다.  
이를 위해 중요한 **음향학적 특징**들이 다음과 같습니다.

---

## 📌 1. Pitch (F0)

**정의:**  
Pitch는 사람의 귀가 느끼는 **소리의 높낮이**를 의미하며, 실제 물리적 측정 값은 **기본 주파수(F0, Fundamental Frequency)**로 표현됩니다.

- 단위: Hz (헤르츠)
- 남성 평균: 약 80~150Hz
- 여성 평균: 약 150~300Hz

---

### 🎯 예시: F0 추출 및 시각화 (YIN 알고리즘)

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load("example.wav", sr=22050)
f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)

# 시간축 생성
frames = range(len(f0))
t = librosa.frames_to_time(frames, sr=sr)

# 시각화
plt.figure(figsize=(10, 3))
plt.plot(t, f0, label='F0 (Pitch)', color='red')
plt.title("Pitch (F0) contour")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## 📌 2. Energy

**정의:**  
에너지는 음성 신호의 **세기(강도)**를 나타내며, **말소리의 크기/강조**를 표현합니다.

- 무성음(쉼, 속삭임): 낮은 energy
- 고음량 발화, 감정 강조: 높은 energy

---

### 🎯 예시: 에너지 계산 및 시각화

```python
import numpy as np

S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
energy = np.linalg.norm(S, axis=0)

plt.figure(figsize=(10, 3))
plt.plot(librosa.frames_to_time(range(len(energy)), sr=sr, hop_length=256), energy)
plt.title("Energy contour")
plt.xlabel("Time (s)")
plt.ylabel("Energy")
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## 📌 3. Duration

**정의:**  
음소 또는 음절이 발음되는 **길이(지속 시간)**  
TTS에서는 각 단어/음절/음소가 얼마나 오래 유지되는지를 모델링합니다.

- "사랑해": 사↗랑→해↘ (길이 다름)
- FastSpeech, VITS 등에서 `Duration Predictor`로 사용됨

---

### 🎯 예시: 음소별 Duration 예시 (개념)

| 음소 | 시작시간 (s) | 종료시간 (s) | Duration (ms) |
|------|--------------|--------------|----------------|
| s    | 0.00         | 0.07         | 70             |
| a    | 0.07         | 0.18         | 110            |
| r    | 0.18         | 0.24         | 60             |

> ※ 실제 Duration은 Forced Alignment 도구 (ex. Montreal Forced Aligner) 로 추출
## 🔍 Forced Alignment의 원리

**Forced Alignment (강제 정렬)** 은  
👉 "음성(wav)"과 "문장 텍스트"를 기반으로  
👉 "각 단어·음소가 음성 내에서 어디에 위치하는지" 자동으로 찾아내는 알고리즘입니다.

---

## 📌 기본 아이디어

> 🎧 음성(wav) + 📄 텍스트(label)  
> → 📊 시간축 위에 "이 단어는 언제부터 언제까지"라는 정보 출력

---

## 🧠 핵심 구성요소

| 구성 요소       | 설명                                                                 |
|----------------|----------------------------------------------------------------------|
| 음향 모델       | 각 음소가 어떤 소리 특성을 갖는지 학습한 모델 (예: GMM-HMM, DNN-HMM) |
| 발음 사전       | 단어 → 음소 시퀀스 매핑 (`hello → HH AH L OW`)                       |
| 디코더          | 오디오에 가장 잘 맞는 시간-음소 시퀀스를 찾음 (Viterbi decoding 사용) |

---

## ⚙️ 동작 흐름 (Step-by-Step)

```text
1. "Hello my name is John" 텍스트 입력
2. 발음 사전으로 음소 변환 → ["HH", "AH", "L", "OW", "M", "AY", ...]
3. 오디오를 프레임 단위로 나눔 (~10ms 단위)
4. 음향 모델이 각 프레임의 음소 확률을 예측
5. Viterbi 알고리즘으로 최적의 타이밍 시퀀스를 추론
6. 결과를 TextGrid 등 시간 정보로 출력
```

---

## 🔄 예시: Alignment 예측 원리

| 프레임 시점 (초) | 예측 음소 확률 (예시)        | 가장 가능성 높은 음소 |
|------------------|------------------------------|------------------------|
| 0.00~0.01        | HH: 0.8, AH: 0.1, OW: 0.1     | HH                     |
| 0.01~0.02        | HH: 0.5, AH: 0.4, OW: 0.1     | HH                     |
| 0.02~0.03        | AH: 0.7, OW: 0.2, L: 0.1      | AH                     |
| 0.03~0.04        | AH: 0.4, L: 0.5, OW: 0.1      | L                      |

→ 디코더가 **"HH: 0.00~0.02, AH: 0.02~0.03, L: 0.03~..."** 로 추정

---

## 📚 수학적 원리 요약

- **입력:** 음성 파형 $x = \{x_1, x_2, ..., x_T\}$  
- **목표:** 음소 시퀀스 $y = \{y_1, y_2, ..., y_N\}$ 와 일치하도록  
- **경로 추정:**  
  $$
  \arg\max_{a_1, ..., a_T} \prod_{t=1}^{T} P(x_t \mid a_t) \cdot P(a_t \mid a_{t-1})
  $$
  → HMM + Viterbi로 최적 경로 추론

---

## 🧠 왜 중요한가?

| 활용 | 설명 |
|------|------|
| TTS Duration 학습 | FastSpeech, VITS 등에서 각 음소의 길이를 예측하기 위해 GT duration 필요 |
| 음성 인식 학습 | 입력 음성과 텍스트를 정렬해 supervised 학습 가능 |
| 감정 분석 | 발화 속도/강세/길이 패턴 파악 가능 |
| 평가 지표 | 실제 발화 속도와 정렬 정확도를 수치로 측정 가능 |

---

## 🛠️ 사용 가능한 알고리즘/도구

| 도구               | 특징                             |
|--------------------|----------------------------------|
| Montreal Forced Aligner (MFA) | 가장 널리 쓰임, 정확도 우수, 다국어 지원 |
| Gentle             | 웹 기반, CMUdict 기반, 간편함    |
| Aeneas             | 텍스트-오디오 정렬 (MP3 등)      |
| Prosodylab-Aligner | Kaldi 기반, 고전적인 HMM 방식     |

---

## ✅ 요약

> Forced Alignment은 **음성과 텍스트를 시간축 위에 정렬**해주는 기술로,  
> TTS 모델의 자연스러운 타이밍 학습에 필수적인 기반 데이터를 제공합니다.
## ⏱️ Forced Alignment & Duration 추출

**Forced Alignment (강제 정렬)** 은 **오디오**와 **문장 텍스트**를 입력으로 받아  
**각 음소/단어가 언제 시작하고 끝나는지 시간 정보를 추출하는 작업**입니다.

TTS에서 **Duration**은 이 강제 정렬을 통해 추출합니다.

---

### 🛠️ 대표 도구: Montreal Forced Aligner (MFA)

- 오디오와 정답 텍스트를 정렬하여 **음소/음절 단위 duration** 추출
- 다국어 지원 (한국어도 가능)
- Kaldi 기반, 정확도 높음
- 결과를 `TextGrid` 형식으로 출력 (Praat 호환)

---

## 📦 설치 방법

1. 공식 홈페이지에서 바이너리 설치  
   👉 https://montreal-forced-aligner.readthedocs.io/

2. 예시 (Linux / Mac):

```bash
conda create -n aligner python=3.8
conda activate aligner
pip install montreal-forced-aligner
```

3. 모델 다운로드

```bash
mfa model download acoustic english
mfa model download dictionary english
```

---

## 📁 폴더 구조 예시

```
alignment_data/
├── wav/                 ← 오디오 파일들 (.wav)
│   └── hello.wav
├── lab/                 ← 텍스트 파일들 (.lab or .txt, 오디오와 동일 이름)
│   └── hello.lab        (내용: Hello, my name is John.)
```

---

## ▶️ 실행 명령어

```bash
mfa align wav/ lab/ english_mfa english_out/
```

- `wav/`: 오디오 경로
- `lab/`: 텍스트 경로
- `english_mfa`: 음향 모델
- `english_out/`: 결과 저장 폴더

---

## 📄 출력 예시: `hello.TextGrid`

```
IntervalTier: words
"Hello"   0.00 - 0.48
"my"      0.48 - 0.65
"name"    0.65 - 0.93
"is"      0.93 - 1.05
"John"    1.05 - 1.42

IntervalTier: phones
"HH"      0.00 - 0.05
"AH"      0.05 - 0.18
"L"       0.18 - 0.30
"OW"      0.30 - 0.48
...
```

### 🎯 의미
| 단어   | 시작(s) | 끝(s) | Duration(ms) |
|--------|---------|--------|----------------|
| Hello  | 0.00    | 0.48   | 480            |
| John   | 1.05    | 1.42   | 370            |

---

## 🧪 Python으로 TextGrid 파싱하기

```python
# pip install textgrid
from textgrid import TextGrid

tg = TextGrid.fromFile("english_out/hello.TextGrid")

word_tier = tg.getFirst("words")
for interval in word_tier:
    print(f"{interval.mark}: {interval.minTime:.2f}s → {interval.maxTime:.2f}s")
```

---

## 📌 요약

| 항목                | 설명                                     |
|---------------------|------------------------------------------|
| 입력                | 오디오 파일 (.wav) + 텍스트 파일 (.lab)  |
| 출력                | 각 음소/단어의 시작~끝 시간 (`TextGrid`) |
| 활용                | TTS의 Duration predictor 학습 등         |
| 기타 도구           | `Aeneas`, `Gentle`, `Prosodylab-aligner` 등 |

---

## ✅ 활용 예

- FastSpeech: 텍스트 → duration → 음향
- VITS: duration 제어 가능한 expressive TTS
- 학습된 alignment으로 attention-free 모델 구성 가능


---

## 📌 4. Formant (포먼트)

**정의:**  
**Formant**는 사람의 발성 기관(입, 코, 혀 등) 구조에 의해 형성되는 **공명 주파수**입니다.

- F1, F2, F3 … 로 나타냄
- 모음 구분에 매우 중요 (예: "아" vs "이")

---

### 🎯 예시: Formant 추정 (Praat, parselmouth 이용)

```python
# pip install praat-parselmouth
import parselmouth
snd = parselmouth.Sound("example.wav")
formant = snd.to_formant_burg()

times = [t for t in np.linspace(0, snd.duration, 100)]
f1 = [formant.get_value_at_time(1, t) for t in times]
f2 = [formant.get_value_at_time(2, t) for t in times]

plt.figure(figsize=(10, 3))
plt.plot(times, f1, label='F1')
plt.plot(times, f2, label='F2')
plt.title("Formant frequencies over time")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## 📌 5. Prosody (운율)

**정의:**  
Prosody는 **강세, 높낮이, 말의 빠르기** 등 **리듬감 있는 말투**를 의미합니다.  
문장 억양, 감정 표현, 질문/명령 구분 등에 핵심적 역할.

---

### 🎯 예시: Prosody에 따른 의미 변화

| 문장             | 운율 (Prosody)                              | 의미                                 |
|------------------|---------------------------------------------|--------------------------------------|
| 사랑해.          | 평탄하게 → 낮은 F0, 일정한 duration         | 담담한 고백                          |
| 사랑해↗          | 점점 올라감 → 상승 F0, 짧은 duration         | 애교스럽거나 놀란 감정               |
| 사↘랑↗해↘        | 감정 강조 → pitch up/down, duration 길게     | 극적이고 강한 표현                   |

---

## 🧠 요약

| 개념     | 설명                       | 활용 예시                    |
|----------|----------------------------|------------------------------|
| Pitch    | 음의 높낮이 (Hz)            | 억양, 질문/평서 구분         |
| Energy   | 말의 세기/강조              | 감정, 속삭임, 소리 크기 표현 |
| Duration | 각 음절의 길이 (ms)         | 빠르게/천천히 말하기         |
| Formant  | 공명 주파수 (F1, F2...)     | 모음 구분, 음질 특성         |
| Prosody  | 운율: 높낮이 + 길이 + 강세  | 감정 표현, 자연스러운 말투   |

---

이러한 요소는 TTS의 **자연스러움, 표현력, 감정**을 크게 좌우합니다.  
고급 TTS 모델은 이들 특성을 **explicit하게 제어**하거나 **모델이 자동 학습**하도록 설계됩니다.
