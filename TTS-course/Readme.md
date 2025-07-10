```markdown
# 🗣️ Text-to-Speech (TTS) 학습 로드맵

---

## TTS 학습 개요

TTS를 배우기 위해서는 다음과 같은 5단계로 나눠 학습하는 것이 효과적입니다:
📂 TTS 학습 로드맵
│
├── 📁 1. 기초 개념
│   ├── 📄 텍스트 처리
│   │   ├── Text Normalization (숫자/기호/약어 처리)
│   │   └── Tokenization (단어 → 자소/음소/음절 단위 분해)
│   ├── 📄 음성 기초
│   │   ├── Waveform (파형, sampling rate)
│   │   ├── STFT / Spectrogram / Mel-Spectrogram
│   │   └── 시각화: librosa, matplotlib
│   ├── 📄 음향학 개념
│   │   ├── Pitch (F0), Energy, Duration
│   │   └── Formant, Prosody (운율 제어)
│   └── 📄 딥러닝 기초
│       ├── CNN / RNN / Transformer 이해
│       ├── Optimizer (Adam, AdamW)
│       └── Loss (L1/L2, GAN Loss, Duration Loss 등)
│
├── 📁 2. TTS 기본 구조
│   ├── 📄 Stage 1: Text → Acoustic Feature
│   │   ├── Tacotron 1/2 (seq2seq + attention)
│   │   └── FastSpeech 1/2 (non-autoregressive + duration)
│   ├── 📄 Stage 2: Acoustic → Waveform
│   │   ├── Griffin-Lim (기초적)
│   │   ├── WaveNet, WaveGlow
│   │   └── HiFi-GAN, UnivNet (고속 + 고품질)
│   └── 📄 Stage 3: End-to-End 모델
│       ├── VITS (VAE + Flow + GAN 통합형)
│       ├── Glow-TTS (flow 기반)
│       └── Grad-TTS, DiffTTS (diffusion 기반)
│
├── 📁 3. 고급 기술
│   ├── 📄 음색 복제 (Voice Cloning)
│   │   ├── Speaker Embedding, x-vector
│   │   └── YourTTS, SV2TTS, Meta-TTS
│   ├── 📄 감정/스타일 조절 (Style Transfer)
│   │   ├── GST (Global Style Token)
│   │   └── StyleTTS, StyleTTS2
│   ├── 📄 멀티링궐 / 제어형 TTS
│   │   ├── Prompt 기반 제어 (Bark 등)
│   │   └── Multilingual Speech (mBART, multilingual VITS)
│   └── 📄 최신 트렌드
│       ├── Bark (GPT + Audio)
│       ├── CosyVoice, StyleTTS2
│       └── Spark-TTS (LLM 기반 고품질)
│
├── 📁 4. 실습 및 배포
│   ├── 📄 실시간 TTS / Streaming (low-latency inference, WebSocket 등)
│   ├── 📄 음성 웹 데모 (Streamlit / Gradio / FastAPI)
│   └── 📄 학습 파이프라인 구축 (데이터 → 전처리 → 학습 → 추론)
├── 📁 5. 확장 영역
│   ├── 📄 평가 지표 (Evaluation)
│   │   ├── MOS (Mean Opinion Score)
│   │   ├── MCD (Mel Cepstral Distortion)
│   │   └── CER/WER (음성→문자 전환 후 평가)
│   ├── 📄 음성 기반 멀티모달 학습
│   │   ├── Text-to-Audio (e.g., AudioLDM, MusicLM)
│   │   └── Text-to-Video + Speech (Bark + VideoGPT)
│   ├── 📄 퍼스널 TTS 서비스 구축
│   │   ├── 사용자 목소리 1분 fine-tuning
│   │   └── Voice App 만들기 (FastAPI + TTS)
│   └── 📄 연구 주제/논문 읽기
│       ├── End-to-End TTS 최신 동향 (ICASSP, Interspeech)
│       ├── TTS + Diffusion + GAN 융합 연구
│       └── Multilingual / Code-Switching / Expressive TTS
```
# 🗣️ TTS 학습 로드맵 개요

이 로드맵은 Text-to-Speech (TTS) 기술을 체계적으로 배우기 위한 다섯 개의 단계로 구성되어 있으며, 각 단계는 실무와 연구에 필요한 이론 및 기술을 균형 있게 포함합니다.

---

## 📌 단계별 오버뷰

### 1. 기초 개념
TTS의 기초가 되는 텍스트 처리, 음성 신호, 음향학, 딥러닝 기법을 학습합니다.  
텍스트 → 음소 분해, 음성 → 스펙트로그램 이해, CNN/RNN/Transformer 구조 및 Loss 개념을 다룹니다.

### 2. TTS 기본 구조
TTS 파이프라인의 3단계 구조를 학습합니다:
- Stage 1: 텍스트 → 음향 특성 (Tacotron, FastSpeech)
- Stage 2: 음향 특성 → 파형 (Griffin-Lim, WaveNet, HiFi-GAN 등)
- Stage 3: End-to-End 모델 (VITS, Glow-TTS, DiffTTS 등)

### 3. 고급 기술
고급 TTS 기술을 통해 실제 응용이 가능한 시스템을 학습합니다:
- 음색 복제 (Voice Cloning)
- 감정/스타일 제어
- 멀티링궐 / 제어형 TTS
- 최신 트렌드: Bark, CosyVoice, Spark-TTS 등

### 4. 실습 및 배포
TTS 모델을 실제 환경에서 활용하는 기술을 익힙니다:
- 실시간 스트리밍 추론 구현
- Gradio/Streamlit 기반 데모 제작
- 학습 파이프라인 구축 (데이터 → 모델 → 추론)

### 5. 확장 영역
TTS 연구 및 응용 분야 확장:
- 평가 지표 이해 및 적용 (MOS, MCD, CER/WER)
- TTS + 멀티모달 학습 (Text-to-Audio/Video)
- 사용자 맞춤형 음성 서비스 구축
- 최신 논문 리뷰 및 연구 주제 탐색

---

> ✅ 본 로드맵은 초보자부터 실무자, 연구자까지 모두를 위한 TTS 학습 여정을 제공합니다.

