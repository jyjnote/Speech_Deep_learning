import gradio as gr
import torch
import numpy as np
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.common import set_all_random_seed
import sys
import os
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.utils.file_utils import load_wav

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
    for word in text.strip().split():
        buffer += word + " "
        for out in cosyvoice.inference_cross_lingual(buffer, prompt_speech, stream=True):
            wav = out['tts_speech'].numpy().flatten()
            yield (cosyvoice.sample_rate, wav)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("### 🌍 실시간 교차 언어 복제 TTS (CosyVoice2 Streaming Mode)")
    text_input = gr.Textbox(label="텍스트 입력", placeholder="단어를 입력하면 바로 음성이 재생됩니다")
    seed = gr.Number(value=1234, label="랜덤 시드")
    audio_out = gr.Audio(label="🎧 합성 오디오", autoplay=True, streaming=True)

    text_input.change(stream_tts_cross_lingual, inputs=[text_input, seed], outputs=[audio_out])

demo.launch(server_name='0.0.0.0', server_port=8888)
