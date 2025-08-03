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

# 고정 prompt 음성 파일
prompt_path = '/mnt/raid0/jjy/CosyVoice/asset/zero_shot_prompt.wav'
prompt_sr = 16000

if not os.path.exists(prompt_path):
    raise FileNotFoundError(f"Prompt 파일이 존재하지 않습니다: {prompt_path}")
prompt_speech = load_wav(prompt_path, prompt_sr)
prompt_text = "good day"  # 제로샷 복제용 프롬프트 텍스트

# 추론 함수
def stream_tts_mode(text, mode, seed):
    set_all_random_seed(seed)
    buffer = ""
    for word in text.strip().split():
        buffer += word + " "
        if mode == "제로샷 복제":
            for out in cosyvoice.inference_zero_shot(buffer, prompt_text, prompt_speech, stream=True):
                yield (cosyvoice.sample_rate, out['tts_speech'].numpy().flatten())
        elif mode == "교차 언어 복제":
            for out in cosyvoice.inference_cross_lingual(buffer, prompt_speech, stream=True):
                yield (cosyvoice.sample_rate, out['tts_speech'].numpy().flatten())
        else:
            raise ValueError("지원하지 않는 모드입니다.")

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("###  실시간 음색 복제 TTS (CosyVoice2 Streaming)")

    with gr.Row():
        mode_radio = gr.Radio(choices=["제로샷 복제", "교차 언어 복제"], value="제로샷 복제", label="복제 모드 선택")
        seed = gr.Number(value=1234, label="랜덤 시드")

    text_input = gr.Textbox(label="텍스트 입력", placeholder="단어를 입력하면 바로 음성이 재생됩니다")
    audio_out = gr.Audio(label="🎧 합성 오디오", autoplay=True, streaming=True)

    text_input.change(stream_tts_mode, inputs=[text_input, mode_radio, seed], outputs=[audio_out])

demo.launch(server_name='0.0.0.0', server_port=8888)
