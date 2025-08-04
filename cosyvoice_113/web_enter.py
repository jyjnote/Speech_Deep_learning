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

# argparse로 포트 인자 받기
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

# 실시간 스트리밍 TTS 함수
def stream_tts_cross_lingual(text, seed):
    set_all_random_seed(seed)
    buffer = ""
    log = ""

    for word in text.strip().split():
        buffer += word + " "
        log += f"[INFO] 입력된 단어: '{word}' → 문맥: \"{buffer.strip()}\"\n"

        for i, out in enumerate(cosyvoice.inference_cross_lingual(buffer, prompt_speech, stream=True)):
            wav = out['tts_speech'].numpy().flatten()
            log += f"[DEBUG] 🔊 스트리밍 {i+1} 완료 (길이: {len(wav)} 샘플)\n"
            yield (cosyvoice.sample_rate, wav), log

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🌍 실시간 교차 언어 복제 TTS (CosyVoice2 Streaming Mode)")
    
    with gr.Row():
        text_input = gr.Textbox(
            label="텍스트 입력 (엔터 입력 시 음성 생성)",
            placeholder="예: 안녕하세요 만나서 반가워요!",
            lines=1
        )
        seed = gr.Number(value=1234, label="랜덤 시드")

    audio_out = gr.Audio(label="🎧 생성된 음성", autoplay=True, streaming=True)
    log_box = gr.Textbox(label="📋 디버깅 로그", lines=10, interactive=False)

    # 제출 시 오디오, 로그 함께 반환
    text_input.submit(fn=stream_tts_cross_lingual, inputs=[text_input, seed], outputs=[audio_out, log_box])

# CLI로 받은 포트에서 실행
demo.launch(server_name='0.0.0.0', server_port=args.port)
