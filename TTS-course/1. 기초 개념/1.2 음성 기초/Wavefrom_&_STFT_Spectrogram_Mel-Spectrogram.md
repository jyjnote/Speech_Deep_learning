## 🎧 음성 기초 (Waveform, Spectrogram, 시각화)

TTS에서 음성을 다루기 위해서는 음성 신호의 기본 표현 방식을 이해하는 것이 필수입니다.


---

## 📌 1. Waveform (파형)

**Waveform**은 시간에 따른 음압(소리의 세기) 변화를 나타내는 **1차원 신호**입니다.  
디지털 오디오에서는 아날로그 음성을 일정 간격으로 **샘플링(sampling)** 하여 저장하며, 이때의 간격은 **Sampling Rate**로 정의됩니다.

---

### ✅ 주요 개념

| 항목            | 설명                                                   | 예시 값 / 설명                         |
|-----------------|--------------------------------------------------------|----------------------------------------|
| Sampling Rate   | 1초에 몇 개의 샘플을 추출했는가 (Hz)                   | `22050Hz` → 1초에 22050개의 수치 저장 |
| Amplitude       | 소리의 세기 (진폭), float 값으로 표현됨                | -1.0 ~ 1.0 범위 (일반적 정규화 기준)   |
| Duration        | 오디오의 전체 재생 시간 (초)                           | 샘플 수 ÷ 샘플링레이트 (예: `44100 / 22050 = 2초`) |
| Mono / Stereo   | 오디오의 채널 수                                       | `Mono`: 1채널, `Stereo`: 2채널        |

---

### 🎯 구체적 예시: 2초짜리 mono 오디오 파일 (`sr=22050Hz`)

```python
import librosa

# 1. 오디오 로딩
y, sr = librosa.load("example.wav", sr=22050)  # sr=샘플링레이트

# 2. 파형 정보 출력
print("✅ Sampling Rate (Hz):", sr)
print("✅ Sample Count:", len(y))
print("✅ Duration (초):", len(y) / sr)
print("✅ Amplitude 범위: [{:.3f}, {:.3f}]".format(y.min(), y.max()))
```

#### 🔎 예시 출력
```
✅ Sampling Rate (Hz): 22050
✅ Sample Count: 44100
✅ Duration (초): 2.0
✅ Amplitude 범위: [-0.37, 0.42]
```

---

### 🖼️ 시각화 예시: Waveform

```python
import matplotlib.pyplot as plt
import librosa.display

plt.figure(figsize=(10, 3))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform of example.wav")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()
```

📌 이 그래프는 시간 축 (x축) 에 따라 소리의 진폭 (y축) 이 어떻게 변화하는지를 보여줍니다.  
소리가 클수록 진폭이 크고, 무음 구간은 거의 0에 가까운 값입니다.

---

### 💡 참고: 오디오 데이터의 실제 구조

| 항목          | 값 예시                             |
|---------------|--------------------------------------|
| `y`           | `numpy.ndarray` (예: `shape=(44100,)`) |
| `sr`          | `int` (예: `22050`)                  |
| 값 예시       | `[0.0123, -0.0205, 0.0187, ..., 0.0012]` |
| 의미          | 1초에 22050개의 음성 샘플이 저장됨    |

---

## 🧠 요약

> Waveform은 시간 도메인 상의 음성 신호입니다.  
> Sampling rate가 높을수록 정밀한 재현이 가능하지만, 데이터 용량도 커집니다.

실제 Waveform은 다음 단계인 Spectrogram으로 변환되어 TTS나 음성 인식 모델의 입력으로 활용됩니다.


---

### 📌 2. STFT / Spectrogram / Mel-Spectrogram

#### ✅ 2-1. STFT (Short-Time Fourier Transform)

- 음성 신호를 짧은 시간 단위로 나누어 **주파수 분석**을 수행
- 시간축 + 주파수축의 2차원 정보 제공

#### ✅ 2-2. Spectrogram

- STFT의 **복소수 결과를 절댓값으로 변환**하여 시각화한 것
- 진한 색일수록 특정 주파수에서 에너지가 강하다는 뜻

#### ✅ 2-3. Mel-Spectrogram

- 인간 청각에 더 유사한 **Mel Scale**을 적용한 Spectrogram
- 음향 모델(TTS, ASR 등)의 입력으로 자주 사용됨

---

#### 🎯 예시 코드: STFT & Mel-Spectrogram 시각화

```python
import numpy as np

# STFT 수행
D = librosa.stft(y, n_fft=1024, hop_length=256)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Mel-Spectrogram
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
mel_db = librosa.power_to_db(mel, ref=np.max)

# 시각화
plt.figure(figsize=(12, 6))

# 1) Spectrogram
plt.subplot(2, 1, 1)
librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title("Spectrogram (STFT)")

# 2) Mel-Spectrogram
plt.subplot(2, 1, 2)
librosa.display.specshow(mel_db, sr=sr, hop_length=256, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel-Spectrogram")

plt.tight_layout()
plt.show()
```

---

### 📌 비교 요약

| 표현 방식           | 차원      | 설명                                                  | 활용 용도                 |
|--------------------|-----------|-------------------------------------------------------|---------------------------|
| Waveform           | 1차원     | 시간에 따른 진폭 값                                   | 음성 신호 원본           |
| Spectrogram (STFT) | 2차원     | 시간 + 주파수                                          | 음성 분석/시각화         |
| Mel-Spectrogram    | 2차원     | Mel-scale 적용한 스펙트로그램                         | TTS, ASR 모델의 입력      |

---

## 🧠 정리

> Waveform은 음성의 원시적 형태이며,  
> Spectrogram은 음성의 **시간-주파수 패턴**을 파악하기 위한 시각적 표현입니다.  
> Mel-Spectrogram은 TTS/ASR에서 널리 쓰이는 **청각 기반 피처**입니다.

---

필요시 `log-Mel`, `MFCC`, `Pitch`, `Energy` 등 추가 설명도 이어서 제공 가능합니다.
