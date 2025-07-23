# 🗣️ CosyVoice2 LJSpeech 학습 전체 파이프라인

> 이 문서는 LJSpeech 데이터셋을 사용하여 CosyVoice2 모델을 학습하기 위한 전체 과정을 정리한 것입니다.

---

## 📂 디렉토리 구성 예시

```
CosyVoice2/
├── tools/
│   ├── prepare_ljspeech.py
│   ├── extract_embedding.py
│   ├── extract_speech_token.py
│   └── make_parquet_list.py
├── data/
│   └── ljspeech/
│       ├── wav.scp
│       ├── text
│       ├── utt2spk
│       ├── spk2utt
│       └── parquet/
├── pretrained_models/
│   └── CosyVoice2-0.5B/
│       ├── campplus.onnx
│       ├── speech_tokenizer_v2.onnx
│       ├── llm.pt
│       ├── flow.pt
│       ├── hift.pt
│       └── CosyVoice-BlankEN/
└── configs/
    └── train_ljspeech.yaml
```

---

## ✅ 단계별 요약표

| Stage | 설명                          | LJSpeech 변경사항                      |
|-------|-------------------------------|----------------------------------------|
| -1    | LibriTTS 다운로드              | ❌ 불필요 (LJSpeech는 수동 다운로드)         |
| 0     | utt2spk, text, wav.scp 생성   | ✅ `prepare_ljspeech.py` 사용           |
| 1     | Speaker embedding 추출        | ✅ `campplus.onnx` 사용, 동일 적용 가능 |
| 2     | Speech token 추출             | ✅ `speech_tokenizer_v2.onnx` 사용     |
| 3     | Parquet 생성 (`.tar` 저장)    | ✅ 동일 방식 사용                       |
| 4     | 학습 리스트 준비              | ✅ `data.list` 복사                    |
| 5     | 모델 학습 (LLM 등)            | ✅ 경로만 조정                         |
| 6     | Checkpoint average            | ✅ 동일 사용 가능                      |
| 7     | 모델 export (JIT, ONNX)       | ✅ 동일 사용 가능                      |

---

## 🔧 Step-by-step 명령어

### ✅ Step 1: utt2spk, text, wav.scp 생성

```bash
python tools/prepare_ljspeech.py \
  --src_dir /mnt/jjy/CosyVoice2/LJSpeech \
  --des_dir data/ljspeech
```

> 생성되는 파일:
> - `data/ljspeech/wav.scp`
> - `data/ljspeech/text`
> - `data/ljspeech/utt2spk`
> - `data/ljspeech/spk2utt`

---

### ✅ Step 2: Speaker Embedding 추출

```bash
python tools/extract_embedding.py \
  --dir data/ljspeech \
  --onnx_path pretrained_models/CosyVoice2-0.5B/campplus.onnx \
  --num_thread 8
```

> 생성되는 파일:
> - `utt2embedding.pt`, `spk2embedding.pt`

---

### ✅ Step 3: Speech Token 추출

```bash
CUDA_VISIBLE_DEVICES=7 python tools/extract_speech_token.py \
  --dir data/ljspeech \
  --onnx_path pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx \
  --num_thread 8
```

> 생성되는 파일:
> - `utt2speech_token.pt`

---

### ✅ Step 4: Parquet 생성

```bash
mkdir -p data/ljspeech/parquet

python tools/make_parquet_list.py \
  --src_dir data/ljspeech \
  --des_dir data/ljspeech/parquet \
  --num_utts_per_parquet 1000 \
  --num_processes 10
```

> 생성되는 파일들:
> - `parquet_000000000.tar`, ...
> - `data.list`, `utt2data.list`, `spk2data.list`

---

### ✅ Step 5: 학습 리스트 복사

```bash
cp data/ljspeech/parquet/data.list data/train.data.list
cp data/ljspeech/parquet/data.list data/dev.data.list
```

---

### ✅ Step 6: LLM 학습 시작 (GPU 7번)

```bash
export PYTHONPATH=$(pwd):$(pwd)/third_party/Matcha-TTS

CUDA_VISIBLE_DEVICES=7 torchrun --nnodes=1 --nproc_per_node=1 \
  --rdzv_id=ljs_train --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
  cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config configs/train_ljspeech.yaml \
  --train_data data/train.data.list \
  --cv_data data/dev.data.list \
  --qwen_pretrain_path pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN \
  --model llm \
  --checkpoint pretrained_models/CosyVoice2-0.5B/llm.pt \
  --model_dir exp/ljspeech/llm \
  --tensorboard_dir tensorboard/ljspeech/llm \
  --ddp.dist_backend nccl \
  --num_workers 2 \
  --prefetch 100 \
  --pin_memory \
  --use_amp
```

> `train_ljspeech.yaml` 예시 설정:
> ```yaml
> dataset:
>   sample_rate: 22050
>   hop_size: 256
>   win_size: 1024
>   n_fft: 1024
>   fmin: 80
>   fmax: 7600
> ```

---

## 📊 Step 7: 로그 기반 시각화 (선택)

```bash
python3 -c "
import matplotlib.pyplot as plt, re
losses, accs, steps = [], [], []
with open('train.log') as f:
    for line in f:
        if 'DEBUG TRAIN Batch' in line:
            m = re.search(r'loss ([\d\.]+) acc ([\d\.]+)', line)
            if m:
                losses.append(float(m.group(1)))
                accs.append(float(m.group(2)))
                steps.append(len(losses))
if losses:
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(steps, losses); plt.title('Loss'); plt.grid()
    plt.subplot(1,2,2); plt.plot(steps, accs); plt.title('Accuracy'); plt.grid()
    plt.tight_layout(); plt.savefig('training_progress.png'); print('✅ saved.')
else:
    print('❌ No matching logs.')
"
```

---

## ✅ 이후 작업

- `flow` 모델 학습
- `hifigan` 학습
- `average_model.py`로 체크포인트 평균
- `export_jit.py`, `export_onnx.py`로 추론용 모델 변환

---
