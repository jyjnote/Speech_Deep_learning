import gradio as gr
import torch
import numpy as np
import argparse
import sys
import os
from omegaconf import OmegaConf

# CUDA_VISIBLE_DEVICES=4 python streaming_web.py --port 50000 --config configs/streaming_config.yaml
# hello everyone good and goodnight all guys
# docker exec -it cosyvoice116 /bin/bash

# 내부 모듈 경로 추가
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.common import set_all_random_seed
from cosyvoice.utils.file_utils import load_wav

# CLI 인자 파싱
parser = argparse.ArgumentParser(description="CosyVoice2 Streaming Web UI")
parser.add_argument("--port", type=int, default=8888, help="Gradio 서버 포트 번호")
parser.add_argument("--config", type=str, default="configs/streaming_config.yaml", help="Streaming 설정 파일 경로")
args = parser.parse_args()

# 설정 로딩
cfg = OmegaConf.load(args.config)
MIN_WORDS = cfg.streaming_tts.min_words
MIN_CHARS = cfg.streaming_tts.min_chars

# CosyVoice2 모델 로딩
model_dir = 'pretrained_models/CosyVoice2-0.5B'
cosyvoice = CosyVoice2(model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False)

# 프롬프트 wav 고정 경로
prompt_path = '/mnt/raid0/jjy/CosyVoice/asset/zero_shot_prompt.wav'
prompt_sr = 16000
if not os.path.exists(prompt_path):
    raise FileNotFoundError(f"Prompt 파일이 존재하지 않습니다: {prompt_path}")
prompt_speech = load_wav(prompt_path, prompt_sr)

# 실시간 TTS 함수
def stream_tts_cross_lingual(text, seed):
    set_all_random_seed(seed)
    logs = ""
    buffer = []
    for word in text.strip().split():
        buffer.append(word)
        phrase = " ".join(buffer)
        if len(buffer) >= MIN_WORDS or len(phrase) >= MIN_CHARS:
            logs += f"▶️ 입력 청크: {phrase}\n"
            for out in cosyvoice.inference_cross_lingual(phrase, prompt_speech, stream=True):
                wav = out['tts_speech'].numpy().flatten()
                logs += f"🔊 음성 생성 완료 (길이: {len(wav)})\n"
                yield (cosyvoice.sample_rate, wav), logs
            buffer = []

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("### 🌍 실시간 교차 언어 복제 TTS (CosyVoice2 Streaming Mode)")

    with gr.Row():
        text_input = gr.Textbox(label="텍스트 입력", placeholder="단어를 입력하면 일정 단위 이상일 때 음성이 재생됩니다")
        seed = gr.Number(value=1234, label="랜덤 시드", precision=0)

    audio_out = gr.Audio(label="🎧 합성 오디오", autoplay=True, streaming=False) ##### False로 바꿀것 매우 강력하게....유력한 후보
    log_display = gr.Textbox(label="🪵 디버깅 로그", lines=10, interactive=False)

    text_input.change(
        fn=stream_tts_cross_lingual,
        inputs=[text_input, seed],
        outputs=[audio_out, log_display]
    )

# 서버 실행
demo.launch(server_name='0.0.0.0', server_port=args.port)
