## 📄 딥러닝 기초 (음성 기반 TTS/ASR 관점)

음성 처리(TTS, ASR 등)에서 자주 쓰이는 **딥러닝 구성 요소**를 정리합니다.

---

## 🧠 1. 딥러닝 구조

### 📌 CNN (Convolutional Neural Network)

- **용도:** 주로 spectrogram의 local pattern 추출에 사용
- **예시:** Tacotron2의 postnet, HiFi-GAN의 discriminator 등

```text
입력: Mel-spectrogram (80 x 1000)
CNN → filter로 local frequency-temporal feature 추출
→ 감정, 발음 강조, 노이즈 제거 등에 효과적
```

---

### 📌 RNN (LSTM, GRU 등)

- **용도:** 시간 순서를 따라 정보를 처리하는 모델
- **예시:** Tacotron1의 decoder, ASR encoder 등

```text
입력: 음소 시퀀스 → [embedding_1, ..., embedding_T]
RNN → 시간 흐름에 따라 상태 업데이트
→ 자연스러운 발음 흐름 예측 가능
```
## 🔤 음소 시퀀스 → 벡터 시퀀스 → 2차원 텐서

### 📌 전제: 음소 하나 = 하나의 벡터 (embedding)

딥러닝 모델에서는 음소나 글자를 그대로 쓰지 않고,  
**각 음소를 고정된 차원의 벡터**로 임베딩합니다.

예를 들어:

| 음소 | 임베딩 벡터 (예시, 4차원)        |
|------|----------------------------------|
| "s"  | [0.1, -0.2, 0.3, 0.7]            |
| "a"  | [0.5, 0.1, -0.4, 0.2]            |
| "k"  | [-0.3, 0.6, 0.0, -0.1]           |

---

### ✅ 따라서 입력 전체는?

**음소 시퀀스 = 벡터 시퀀스 = 2차원 행렬**

```text
예: "s a k i" → 임베딩 → [[0.1, -0.2, 0.3, 0.7],
                          [0.5, 0.1, -0.4, 0.2],
                          [-0.3, 0.6, 0.0, -0.1],
                          [0.2, -0.1, 0.5, 0.3]]
```

→ 이 결과는 shape `(T, D)`의 2차원 텐서  
- `T`: 음소 길이 (time step)  
- `D`: 임베딩 차원 (예: 256, 512 등)

---

### 🔁 이후에는?

이 2D 텐서 (음소 벡터 시퀀스)는 다음과 같이 처리됩니다:

- **RNN 계열**: 시간 순서대로 한 벡터씩 처리 (`t=1 → t=2 → ...`)
- **Transformer**: 벡터들을 동시에 attention 처리 (자기참조)

→ **출력도 마찬가지로 2차원 시퀀스 (T, D')**가 됩니다. 이 시퀀스는  
Mel-spectrogram / acoustic feature / unit sequence / waveform 등에 연결됩니다.

---

### 🧠 요약

| 개념             | 구조                      | 설명                             |
|------------------|---------------------------|----------------------------------|
| 음소 시퀀스       | `[p1, p2, ..., pT]`        | 텍스트 입력                      |
| 임베딩 시퀀스     | `[e1, e2, ..., eT]`        | 각 음소를 벡터로 변환            |
| 텐서 형태         | `Tensor: shape = (T, D)`   | 2차원 텐서 (시간, 차원)          |
| RNN/Transformer 입력 | 이 2차원 텐서를 받아 처리     | 시간 순 또는 병렬 연산 수행       |

---

> 🎯 결론: **음소 시퀀스는 임베딩을 거쳐 2차원 시퀀스 텐서가 되며**,  
> **딥러닝 모델의 시간 처리 입력 형태로 사용됩니다**.
---

### 📌 Transformer

- **용도:** self-attention 기반, 병렬처리와 긴 문맥 이해에 강함
- **예시:** FastSpeech, VITS, Whisper, Wav2Vec2 등 최신 모델 대부분

```text
입력: 음소 시퀀스 / 멜 스펙트로그램
→ Positional Encoding 추가 → Multi-head Attention
→ 긴 문장, 억양 조절, multilingual 처리 등에 탁월
```

---

## ⚙️ 2. Optimizer

### ✅ Adam

- 학습률 자동 조정 (모멘텀 + RMSProp)
- TTS/ASR 거의 모든 모델의 기본 Optimizer

### ✅ AdamW

- `Adam + weight decay` (L2 정규화)
- VITS, FastSpeech2 등 고성능 모델에서 널리 사용

## 📐 노름2 (L2 Norm, Euclidean Norm) 이해

---

### 📌 정의

**L2 노름**은 벡터의 "길이" 또는 "거리"를 측정하는 방식입니다.  
두 벡터가 얼마나 가까운지, 예측값이 정답과 얼마나 다른지를 측정할 때 사용됩니다.

수식:

$$
\|x - y\|_2 = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + \cdots + (x_n - y_n)^2}
$$

---

## 🧠 왜 사용하나요?

| 이유 | 설명 |
|------|------|
| 📉 예측 오차 계산 | 모델이 출력한 벡터와 정답 벡터 간의 거리 측정 |
| 🔄 연속적 수치 예측 | 멜스펙트로그램, 음성 파형 등에서 연속값을 다룸 |
| 📏 수학적으로 해석 용이 | 미분 가능하고 convex라서 최적화에 유리 |

---

## 🎯 예시: L2 노름으로 거리 계산

### 🎲 데이터

```text
Target vector (정답):     y  = [2.0, 3.0, 1.0]
Predicted vector (예측):  ŷ = [2.5, 2.0, 0.0]
```

### 🧮 계산

```text
차이 벡터: [2.5 - 2.0, 2.0 - 3.0, 0.0 - 1.0] = [0.5, -1.0, -1.0]

제곱합: 0.5² + (-1.0)² + (-1.0)² = 0.25 + 1 + 1 = 2.25

L2 노름: sqrt(2.25) ≈ 1.5
```

---

## 🆚 L1 노름과의 비교

| 항목         | L1 노름                     | L2 노름                     |
|--------------|------------------------------|------------------------------|
| 수식         | $\|x - y\|_1 = \sum |x_i - y_i|$ $| $\|x - y\|_2 = \sqrt{\sum (x_i - y_i)^2}$ |
| 부드러움     | 🔹 sharp (스파스 유도)       | 🔸 smooth (작은 오차에 민감) |
| 사용 예시    | 텍스트 임베딩, 정규화 등     | 멜스펙트로그램 예측, 회귀   |
---

## 📊 PyTorch 예시

```python
import torch
import torch.nn as nn

# 정답과 예측 벡터
y = torch.tensor([2.0, 3.0, 1.0])
y_hat = torch.tensor([2.5, 2.0, 0.0])

# L2 Loss (MSE)
mse_loss = nn.MSELoss()
loss = mse_loss(y_hat, y)

print("L2 Loss (MSE):", loss.item())  # 출력: 0.75
```

🧮 왜 0.75일까?

```
MSE = ((0.5)^2 + (-1)^2 + (-1)^2) / 3 = 2.25 / 3 = 0.75
```

---

## 📌 요약

| 개념        | 설명                                 |
|-------------|--------------------------------------|
| L2 노름     | 예측값과 정답 사이의 거리 (제곱의 평균) |
| 사용 이유   | 부드러운 회귀, 미분 가능, 수학적 안정성 |
| 사용 예시   | TTS의 mel 예측, waveform 회귀, pitch 등 |

> ✅ L2 노름은 **예측값이 정답과 얼마나 멀리 떨어져 있는지**를 연속적 수치로 정밀하게 측정합니다.

```python
import torch.optim as optim
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
```

---

## 🎯 3. Loss Functions (음성 모델 중심)

| Loss 종류        | 설명                                           | 적용 예시                                |
|------------------|------------------------------------------------|-------------------------------------------|
| **L1 Loss**      | MAE (Mean Absolute Error), 부드러운 예측에 유리 | Mel target과 예측값 차이 (`|y - ŷ|`)      |
| **L2 Loss**      | MSE (Mean Squared Error), 노이즈 민감함        | 음성 에너지, f0 예측 등                   |
| **GAN Loss**     | 생성자-판별자 경쟁 (HiFi-GAN, VITS 등)         | 고품질 waveform 생성                      |
| **Duration Loss**| 예측 duration과 forced-alignment 기반 GT 비교 | FastSpeech의 duration predictor 학습      |
| **CTC Loss**     | 비정렬 입력-출력 정렬에 적합 (ASR 전용)        | ASR (Wav2Vec2, Whisper)                   |

---

## 🧪 CTC Loss란?

> CTC (Connectionist Temporal Classification) Loss는  
> 🔁 입력과 출력 시퀀스가 길이도 다르고 정렬도 안 돼 있을 때 사용하는 음성 인식 전용 손실 함수입니다.

### ✨ 예시

```text
입력 오디오 길이: 100프레임
예측 시퀀스: ["_", "_", "h", "e", "_", "l", "l", "o", "_", "_"]
Target 텍스트: ["h", "e", "l", "o"]

→ "_"는 blank token, 중복은 collapse됨
→ "h", "e", "l", "l", "o" → "hello"
→ GT와 정렬 없이 loss 계산 가능
```

## 📌 GT (Ground Truth)란?

**GT**는 **Ground Truth**의 약자로,  
👉 모델이 학습할 때 사용하는 **정답(label)** 또는 **참값(reference)**을 의미합니다.

---

## 🎯 예시별 GT의 의미

| 작업 유형              | GT 의미                                     |
|------------------------|---------------------------------------------|
| ASR (음성 인식)        | 오디오에 대응되는 정답 텍스트 (`"hello"`)     |
| TTS (음성 합성)        | 텍스트에 대응하는 Mel-spectrogram            |
| Duration Prediction    | Forced Alignment으로 추출한 음소별 길이      |
| Pitch / Energy 예측    | 실제 음성에서 추출된 F0, Energy 값           |
| GAN 기반 음성 합성     | 고음질 waveform (real audio)                 |

---

## 📦 예시 1: TTS에서 GT의 사용

```text
입력 텍스트: "사랑해"

음소 시퀀스: ㅅ ㅏ ㄹ ㅏ ㅇ ㅎ ㅐ

GT Mel-Spectrogram: (80 x T) 크기의 벡터 시퀀스
→ 실제 사람이 녹음한 음성에서 추출한 Mel

모델 출력과 GT mel을 비교하여 L1/L2 Loss 계산
```

---

## 📦 예시 2: Duration Prediction에서 GT

```text
Forced Alignment 결과 (GT):
  - "사": 0.00 ~ 0.12s
  - "랑": 0.12 ~ 0.38s
  - "해": 0.38 ~ 0.60s

GT Duration = [0.12, 0.26, 0.22]  (초 단위)

모델이 예측한 duration과 GT를 비교하여 Duration Loss 계산
```

---

## 📊 사용 목적

| 목적             | 설명                                                |
|------------------|-----------------------------------------------------|
| 학습 Supervison  | 모델이 예측한 값이 GT에 얼마나 가까운지를 학습      |
| Loss 계산        | `Loss(pred, GT)` 형태로 사용                        |
| 평가 지표 계산   | 예: CER, WER, MCD 등 GT와 비교하여 정확도 측정       |
## 🧠 요약

- **GT = Ground Truth = 정답**
- 모델 학습에 필요한 참값 데이터
- 음성에서는 **텍스트, 멜, duration, pitch 등**이 모두 GT가 될 수 있음

> 📌 GT는 모델이 "정답이 뭔지"를 아는 유일한 기준이며,  
> 딥러닝 학습의 출발점입니다.

### 📌 핵심 특징

- 입력 길이 ≠ 출력 길이일 때 학습 가능
- Alignment 없이 훈련 가능 (ASR 전용)
- TTS에서는 거의 쓰지 않지만, **ASR 학습 필수 요소**

---

## 📊 음성 평가 지표 약자 설명 (CER, WER, MCD)

---

### 🅰️ 1. CER: **Character Error Rate**

- **약자 의미:** *Character Error Rate*
- **정의:** 문자 단위 오류 비율  
  $$
  \text{CER} = \frac{S + D + I}{N}
  $$
  - S: 대체(Substitution)
  - D: 삭제(Deletion)
  - I: 삽입(Insertion)
  - N: 전체 GT 문자 수

- **사용 분야:** 음성 인식 (ASR), TTS → STT 평가 등

#### ✅ 예시:
```text
GT 텍스트:    사랑해
예측 텍스트: 사량해  (→ '랑'이 '량'으로 대체됨)

S=1, D=0, I=0 → CER = 1 / 3 = 33.3%
```

---

### 🅱️ 2. WER: **Word Error Rate**

- **약자 의미:** *Word Error Rate*
- **정의:** 단어 단위 오류 비율  
  $$
  \text{WER} = \frac{S + D + I}{N}
  $$

- **사용 분야:** 음성 인식 모델의 핵심 평가 지표

#### ✅ 예시:
```text
GT:    I love you
Pred:  I like you  (→ "love" → "like" 대체)

S=1, D=0, I=0 → WER = 1 / 3 = 33.3%
```

---

### 🆎 3. MCD: **Mel-Cepstral Distortion**

- **약자 의미:** *Mel-Cepstral Distortion*
- **정의:** 생성된 음성과 GT 음성의 **Mel-Cepstrum 간 거리**를 측정하는 수치  
  (보통 음질 유사도를 평가)

- **사용 분야:** TTS 음성 품질 평가 (Waveform or Mel 기반)

#### ✅ 예시 개념:
```text
GT mel vector:     [1.0, 2.0, 3.0, 4.0]
Predicted mel:     [0.8, 2.1, 3.2, 4.3]

→ 벡터 거리 기반 평균 왜곡률 계산 → MCD = 3.01 (dB 단위)

※ 값이 낮을수록 GT와 유사 (좋은 품질)
```

---

## 🧠 요약

| 지표 | 약자 의미               | 기준 단위  | 사용 분야        | 해석                                |
|------|--------------------------|-------------|-------------------|-------------------------------------|
| CER  | Character Error Rate     | 문자 단위  | STT / TTS 평가    | 낮을수록 좋음                       |
| WER  | Word Error Rate          | 단어 단위  | STT 평가          | 낮을수록 좋음                       |
| MCD  | Mel-Cepstral Distortion  | 스펙트럼 거리 | TTS 음질 평가     | 낮을수록 자연스러운 음성           |

> 🎯 CER/WER는 **텍스트 기반 평가**,  
> MCD는 **음성 스펙트럼 기반 평가**입니다.

---


## 🎯 Loss 함수별 PyTorch 예시

```python
import torch.nn as nn

# 기본 TTS loss
l1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()

# CTC Loss (ASR용)
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

# 예시
pred = torch.randn(100, 32, 50).log_softmax(2)  # (T, N, C)
target = torch.randint(1, 50, (32, 20))         # (N, S)
input_lengths = torch.full((32,), 100, dtype=torch.long)
target_lengths = torch.full((32,), 20, dtype=torch.long)

loss = ctc_loss(pred, target, input_lengths, target_lengths)
```

---

## 🧠 정리

| 구분     | 음성 TTS                     | 음성 인식 (ASR)              |
|----------|------------------------------|-------------------------------|
| 주요 구조 | CNN, Transformer, RNN        | CNN + RNN + Transformer       |
| 주요 Loss | L1, L2, GAN, Duration        | CTC, CrossEntropy             |
| 주로 예측 | mel, waveform, pitch, duration | 문자 시퀀스 (음소 or 문자)    |

---

> 🎤 TTS 모델에서는 **L1 + GAN + Duration Loss**가 주로 사용되고  
> 🧠 ASR에서는 **CTC 또는 CrossEntropy + Attention 기반**이 널리 사용됩니다.

필요시 각 loss가 실제 학습에 어떻게 조합되는지 schematic diagram 도 제공 가능합니다.
