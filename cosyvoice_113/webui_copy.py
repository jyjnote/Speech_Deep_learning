# /mnt/raid0/jjy/CosyVoice/streaming/realtime_webui.py

import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import random
import time

# ✅ Matcha-TTS 경로 명확하게 추가
sys.path.append("/mnt/raid0/jjy/CosyVoice/third_party/Matcha-TTS")

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

# -------------------------
# Global setup
# -------------------------
prompt_sr = 16000
prompt_path = "/mnt/raid0/jjy/CosyVoice/asset/zero_shot_prompt.wav"
cosyvoice = None

# -------------------------
# Streaming TTS Handler
# -------------------------
def stream_input_handler(current_text, seed):
    if not current_text.strip():
        logging.debug("[stream_input_handler] Empty input received.")
        return None

    last_word = current_text.strip().split()[-1]  # or use full string
    logging.info(f"[stream_input_handler] Processing word: '{last_word}' with seed {seed}")

    prompt_speech_16k = load_wav(prompt_path, target_sr=prompt_sr)  # 🔧 필수 인자 target_sr 추가
    logging.debug(f"[stream_input_handler] Loaded prompt speech from {prompt_path} at {prompt_sr}Hz")

    set_all_random_seed(seed)

    chunk_idx = 0
    start_time = time.time()

    for output in cosyvoice.inference_zero_shot(
        tts_text=last_word,
        prompt_text=last_word,
        prompt_speech_16k=prompt_speech_16k,
        stream=True
    ):
        sr = cosyvoice.sample_rate
        chunk_len = output['tts_speech'].shape[1] / sr
        chunk_rtf = (time.time() - start_time) / chunk_len
        logging.info(f"[stream_input_handler] 🔊 Chunk {chunk_idx}: {chunk_len:.2f}s, RTF={chunk_rtf:.3f}")
        chunk_idx += 1
        start_time = time.time()
        yield (sr, output['tts_speech'].numpy().flatten())

# -------------------------
# Launch UI
# -------------------------
def launch_demo():
    logging.info("[launch_demo] Starting CosyVoice2 Streaming Gradio UI")
    with gr.Blocks() as demo:
        gr.Markdown("## 💬 CosyVoice2 Real-Time Typing TTS Demo")

        with gr.Row():
            textbox = gr.Textbox(
                label="💡 Type and hear speech immediately",
                lines=1,
                placeholder="Start typing to speak...",
                interactive=True
            )
            seed = gr.Number(value=random.randint(1, 999999), label="Seed", precision=0)

        audio_output = gr.Audio(
            label="🔊 Generated Audio",
            autoplay=True,
            streaming=True,
            type="numpy",
            format="wav"
        )

        textbox.input(
            fn=stream_input_handler,
            inputs=[textbox, seed],
            outputs=[audio_output],
            trigger_mode="always_last",
            show_progress="minimal",
            queue=True
        )

    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=args.port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice2-0.5B')
    args = parser.parse_args()

    logging.info(f"[main] Loading CosyVoice2 model from {args.model_dir}")
    cosyvoice = CosyVoice2(
        model_dir=args.model_dir,
        load_jit=False,
        load_trt=False,
        load_vllm=False
    )
    logging.info("[main] CosyVoice2 model loaded successfully")

    launch_demo()
