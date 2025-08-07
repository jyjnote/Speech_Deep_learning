# /mnt/raid0/jjy/CosyVoice/webui_copy.py

import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import random
import time

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
# Debounced TTS (cross-lingual)
# -------------------------
def generate_tts(current_text, seed):
    if not current_text.strip():
        logging.debug("[generate_tts] Empty input received.")
        return None

    logging.info(f"[generate_tts] Synthesizing: '{current_text}' (seed={seed})")

    prompt_speech_16k = load_wav(prompt_path, target_sr=prompt_sr)
    set_all_random_seed(seed)

    output_gen = cosyvoice.inference_cross_lingual(
        tts_text=current_text,
        prompt_speech_16k=prompt_speech_16k,
        zero_shot_spk_id="",
        stream=False
    )

    output = next(output_gen)
    sr = cosyvoice.sample_rate
    logging.info(f"[generate_tts] ✅ Done. Length: {output['tts_speech'].shape[1] / sr:.2f}s")

    return (sr, output["tts_speech"].numpy().flatten())

# -------------------------
# Launch UI
# -------------------------
def launch_demo():
    logging.info("[launch_demo] Starting CosyVoice2 Debounced Gradio UI")
    with gr.Blocks() as demo:
        gr.Markdown("## ⏱️ CosyVoice2 Typing-to-TTS Demo (Debounced 1s Delay)")

        with gr.Row():
            textbox = gr.Textbox(
                label="💡 Type something...",
                lines=1,
                placeholder="Start typing...",
                interactive=True
            )
            seed = gr.Number(value=random.randint(1, 999999), label="Seed", precision=0)

        audio_output = gr.Audio(
            label="🎧 Synthesized Audio",
            autoplay=True,
            streaming=False,
            type="numpy",
            format="wav"
        )

        # ✅ typing 멈춘 후 1초 후에 TTS 실행
        textbox.change(
            fn=generate_tts,
            inputs=[textbox, seed],
            outputs=audio_output
        )

    demo.launch(server_name="0.0.0.0", server_port=args.port)

# -------------------------
# Entrypoint
# -------------------------
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
