import time
import torch
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav   # 프로젝트 내 로더 (없으면 librosa.load로 대체)

import sys
sys.path.append('third_party/Matcha-TTS')
# 0) 모델 로드
cosyvoice = CosyVoice2(
    model_dir='pretrained_models/CosyVoice2-0.5B',
    load_jit=False, load_trt=False, load_vllm=False, fp16=False
)

# 1) 프롬프트 오디오
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

# 2) "사용자 타이핑"을 흉내내는 제너레이터(영문)
def fake_typing():
    text_pieces = [
        "I just",
        "received a birthday", 
        "gift ",
        "from a dear friend",
        " who lives far away, ",
        "and the unexpected surprise",
        " and heartfelt wishes ",
        "filled my heart with so",
        " much joy ",
        "that my smile bloomed like a flower.",
        "cosyvoice2 실시간 음성 입력 테스트",
        "출력이 한국어도 잘 청크단위로 잘리는지 체크",
        "마",
        "지막 테스트 입력",
        "오디오",
        "출력"
    ]
    for piece in text_pieces:
        yield piece
        time.sleep(0.25)  # typing delay simulation

# 3) 스트리밍 합성 호출 (중요: stream=True)
gen = cosyvoice.inference_zero_shot_typing(
    text_stream=fake_typing(),
    prompt_text="Hope you can outdo me one day!",
    prompt_speech_16k=prompt_speech_16k,
    zero_shot_spk_id="",
    stream=True,
    speed=1.0,
    text_frontend=True,
    chunk_tokens=20,                # 반응 속도↑
    interleave_prompt_in_llm=False
)

# 4) 출력 조각 저장
sr = cosyvoice.sample_rate
for i, out in enumerate(gen, 1):
    wav = out["tts_speech"].cpu()
    torchaudio.save(f"./typing_stream_{i:02d}.wav", wav, sr)
    print("saved:", i, wav.shape)
