import os
import sys
import argparse
import gradio as gr
import torch
import torchaudio
import numpy as np
import librosa
import random

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{ROOT_DIR}/third_party/Matcha-TTS")

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

# -------------------------
# Config
# -------------------------
prompt_sr = 16000
prompt_path = "/mnt/raid0/jjy/CosyVoice/asset/zero_shot_prompt.wav"
max_val = 0.8

# -------------------------
# CLI Args
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice2-0.5B')
args = parser.parse_args()

# -------------------------
# Load Model & Prompt
# -------------------------
cosyvoice = CosyVoice2(args.model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
default_data = np.zeros(cosyvoice.sample_rate)

if os.path.exists(prompt_path):
    prompt_speech_16k = load_wav(prompt_path, prompt_sr)

    # Safe conversion to 2D torch tensor
    if isinstance(prompt_speech_16k, np.ndarray):
        prompt_speech_16k = torch.from_numpy(prompt_speech_16k)

    if prompt_speech_16k.dim() == 1:
        prompt_speech_16k = prompt_speech_16k.unsqueeze(0)

else:
    raise FileNotFoundError(f"Prompt not found: {prompt_path}")

# -------------------------
# Helper
# -------------------------
def postprocess(speech):
    speech, _ = librosa.effects.trim(speech, top_db=60, frame_length=440, hop_length=220)
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    return torch.cat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)

# -------------------------
# Streaming TTS Function
# -------------------------
def stream_tts(text):
    if not text.strip():
        yield (cosyvoice.sample_rate, default_data)
        return

    set_all_random_seed(random.randint(1, int(1e8)))

    print("[DEBUG] Input text:", text)

    # CosyVoice 설정 파라미터 디버깅
    if hasattr(cosyvoice, "chunk_size"):
        print(f"[DEBUG] CosyVoice chunk_size: {cosyvoice.chunk_size}")
    if hasattr(cosyvoice, "num_decoding_left_chunks"):
        print(f"[DEBUG] CosyVoice num_decoding_left_chunks: {cosyvoice.num_decoding_left_chunks}")

    total_tokens = 0
    for i, result in enumerate(cosyvoice.inference_cross_lingual(
        text, prompt_speech_16k, stream=True, speed=1.0
    )):
        num_tokens = result.get("num_tokens", None)
        if num_tokens is not None:
            total_tokens = num_tokens
        else:
            total_tokens += 1  # fallback if not explicitly provided

        print(f"[DEBUG] Step {i}: Generated {total_tokens} tokens so far")

        yield (cosyvoice.sample_rate, result["tts_speech"].numpy().flatten())

# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🗣️ CosyVoice2 실시간 스트리밍 TTS (입력 즉시 음성 생성)")

    with gr.Row():
        input_box = gr.Textbox(label="텍스트 입력", placeholder="입력 시 바로 음성이 생성됩니다", lines=2)

    output_audio = gr.Audio(label="🎧 실시간 출력", streaming=True, autoplay=True)

    # 텍스트 입력 시 자동 음성 생성
    input_box.change(fn=stream_tts, inputs=input_box, outputs=output_audio)

demo.queue()
demo.launch(server_name="0.0.0.0", server_port=args.port)
