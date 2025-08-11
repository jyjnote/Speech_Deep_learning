import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import random
import time

# Matcha-TTS 경로 추가
sys.path.append("/mnt/raid0/jjy/CosyVoice/third_party/Matcha-TTS")

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

# 제로샷 프롬프트 음성 설정
prompt_sr = 16000
prompt_path = "/mnt/raid0/jjy/CosyVoice/asset/zero_shot_prompt.wav"
cosyvoice = None


def generate_tts(current_text, seed):
    current_text = current_text.strip("\n")
    debug_log = ""

    # 입력 검증
    if not current_text.strip():
        msg = "[generate_tts] 입력이 비어있습니다."
        logging.debug(msg)
        return None, msg

    if not current_text.endswith(" "):
        msg = "[generate_tts] 마지막이 공백이 아닙니다. 실행하지 않음."
        logging.info(msg)
        return None, msg

    # 마지막 단어만 추출
    words = current_text.strip().split()
    if not words:
        msg = "[generate_tts] 유효한 단어 없음."
        logging.warning(msg)
        return None, msg

    last_word = words[-1]
    debug_log += f"[generate_tts] Synthesizing last word only: '{last_word}' (seed={seed})\n"

    prompt_speech_16k = load_wav(prompt_path, target_sr=prompt_sr)
    set_all_random_seed(seed)

    try:
        # CosyVoice2 inference
        output_gen = cosyvoice.inference_cross_lingual(
            tts_text=last_word,
            prompt_speech_16k=prompt_speech_16k,
            zero_shot_spk_id="",
            stream=True
        )
        output = next(output_gen)

        sr = cosyvoice.sample_rate
        audio = output["tts_speech"].numpy().flatten()

        tts_tokens = output.get("tts_tokens", None)
        mel = output.get("mel", None)
        token_len = len(tts_tokens) if tts_tokens is not None else -1
        mel_len = mel.shape[-1] if mel is not None else -1

        debug_log += f"샘플레이트: {sr}Hz\n"
        debug_log += f"전체 파형 길이: {len(audio)} samples ({len(audio)/sr:.2f}s)\n"
        debug_log += f"텐서 shape: {output['tts_speech'].shape}\n"
        debug_log += f"토큰 길이: {token_len} | Mel 길이: {mel_len}\n"
        debug_log += "보조 문장 없이 전체 음성 사용\n"
        debug_log += "음성 생성 완료."

        logging.info(debug_log.strip())
        return (sr, audio), debug_log

    except Exception as e:
        debug_log += f"에러 발생: {str(e)}"
        logging.error(debug_log.strip())
        return None, debug_log


def launch_demo():
    logging.info("[launch_demo] Starting CosyVoice2 Gradio UI")

    with gr.Blocks() as demo:
        gr.Markdown("## CosyVoice2: Type-and-speak by word (space-delimited)")
        gr.Markdown("단어 입력 후 스페이스를 누르면 해당 단어만 음성으로 합성됩니다.")

        with gr.Row():
            textbox = gr.Textbox(
                label="Type here...",
                lines=1,
                placeholder="Type words and press space after each",
                interactive=True
            )
            seed = gr.Number(
                value=random.randint(1, 999999),
                label="Seed",
                precision=0
            )

        with gr.Row():
            audio_output = gr.Audio(
                label="Synthesized Audio (only last word)",
                autoplay=True,
                streaming=True,
                type="numpy",
                format="wav"
            )

        debug_output = gr.Textbox(
            label="Debug Log",
            lines=10,
            max_lines=20,
            interactive=False
        )

        # 직접적으로 debounce 로직 없이 기본 함수만 연결
        textbox.change(
            fn=generate_tts,
            inputs=[textbox, seed],
            outputs=[audio_output, debug_output]
        )

    # Gradio queue 설정 적용: concurrency 제한
    demo.queue(max_size=10, default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=50000, help="Gradio UI 포트 번호")
    parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice2-0.5B', help="CosyVoice2 모델 경로")
    args = parser.parse_args()

    # 모델 로딩
    logging.info(f"[main] Loading CosyVoice2 model from {args.model_dir}")
    cosyvoice = CosyVoice2(
        model_dir=args.model_dir,
        load_jit=False,
        load_trt=False,
        load_vllm=False
    )
    logging.info("[main] CosyVoice2 model loaded successfully")

    # UI 실행
    launch_demo()
