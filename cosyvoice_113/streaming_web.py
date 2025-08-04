import gradio as gr
import torch
import numpy as np
import argparse
import sys
import os

# 내부 모듈 경로 추가
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.common import set_all_random_seed
from cosyvoice.utils.file_utils import load_wav

# CLI 인자 파싱 (포트 번호 등)
parser = argparse.ArgumentParser(description="CosyVoice2 Streaming Web UI")
parser.add_argument("--port", type=int, default=8888, help="Gradio 서버 포트 번호")
args = parser.parse_args()

# CosyVoice2 모델 로딩
model_dir = 'pretrained_models/CosyVoice2-0.5B'
cosyvoice = CosyVoice2(model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False)

# 프롬프트 wav 고정 경로
prompt_path = '/mnt/raid0/jjy/CosyVoice/asset/zero_shot_prompt.wav'
prompt_sr = 16000

# 사전 로딩 (속도 개선용)
if not os.path.exists(prompt_path):
    raise FileNotFoundError(f"Prompt 파일이 존재하지 않습니다: {prompt_path}")
prompt_speech = load_wav(prompt_path, prompt_sr)

# 실시간 스트리밍 TTS 함수 (로그 포함)
def stream_tts_cross_lingual(text, seed):
    set_all_random_seed(seed)
    buffer = ""
    logs = ""
    for word in text.strip().split():
        buffer += word + " "
        logs += f"▶️ 입력 단어: {word}\n"
        for out in cosyvoice.inference_cross_lingual(buffer, prompt_speech, stream=True):
            wav = out['tts_speech'].numpy().flatten()
            logs += f"🔊 음성 생성 완료 (길이: {len(wav)})\n"
            yield [(cosyvoice.sample_rate, wav), logs]

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("### 🌍 실시간 교차 언어 복제 TTS (CosyVoice2 Streaming Mode)")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="텍스트 입력", placeholder="단어를 입력하면 바로 음성이 재생됩니다")
            seed = gr.Number(value=1234, label="랜덤 시드")
        with gr.Column():
            audio_out = gr.Audio(label="🎧 합성 오디오", autoplay=True, streaming=True)
            debug_box = gr.Textbox(label="📋 디버그 로그", lines=15, interactive=False)

    # 사용자 입력 변경 시 음성 + 로그 출력
    text_input.change(
        stream_tts_cross_lingual,
        inputs=[text_input, seed],
        outputs=[audio_out, debug_box]
    )

# CLI로 전달받은 포트로 실행
demo.launch(server_name='0.0.0.0', server_port=args.port)
