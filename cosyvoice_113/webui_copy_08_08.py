import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import random
import time

# Matcha-TTS 경로를 sys.path에 추가하여 내부 모듈 import 가능하게 설정
sys.path.append("/mnt/raid0/jjy/CosyVoice/third_party/Matcha-TTS")

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

# 프롬프트 음성 파일 설정 (제로샷 화자 조건용)
prompt_sr = 16000
prompt_path = "/mnt/raid0/jjy/CosyVoice/asset/zero_shot_prompt.wav"
cosyvoice = None  # CosyVoice2 객체는 main에서 초기화


def generate_tts_safe(current_text, seed, is_processing):
    """
    Debounce를 적용한 안전한 음성 합성 함수.
    이전 요청이 처리 중이면 무시하고 상태 메시지만 출력함.

    Returns:
        - 오디오 출력
        - 디버그 메시지
        - is_processing (다음 상태)
    """
    current_text = current_text.strip("\n")

    # 1. 처리 중이면 무시
    if is_processing:
        msg = "[generate_tts_safe] 이전 작업 처리 중... 이번 입력은 무시됩니다."
        logging.debug(msg)
        return None, msg, True

    # 2. 입력이 공백으로 끝나는 경우만 처리
    if not current_text.strip():
        msg = "[generate_tts_safe] 입력이 비어있습니다."
        return None, msg, False
    if not current_text.endswith(" "):
        msg = "[generate_tts_safe] 마지막이 공백이 아닙니다. 실행하지 않음."
        return None, msg, False

    # 3. 마지막 단어 추출
    words = current_text.strip().split()
    if not words:
        msg = "[generate_tts_safe] 유효한 단어 없음."
        return None, msg, False
    last_word = words[-1]

    # 4. 프롬프트 로딩 및 시드 고정
    prompt_speech_16k = load_wav(prompt_path, target_sr=prompt_sr)
    set_all_random_seed(seed)

    debug_log = f"[generate_tts] Synthesizing last word only: '{last_word}' (seed={seed})\n"

    try:
        # 음성 합성
        output_gen = cosyvoice.inference_cross_lingual(
            tts_text=last_word,
            prompt_speech_16k=prompt_speech_16k,
            zero_shot_spk_id="",
            stream=True
        )
        output = next(output_gen)

        sr = cosyvoice.sample_rate
        audio = output["tts_speech"].numpy().flatten()

        # 디버그 출력
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
        return (sr, audio), debug_log, False  # is_processing → False (작업 완료)

    except Exception as e:
        debug_log += f"에러 발생: {str(e)}"
        logging.error(debug_log.strip())
        return None, debug_log, False  # 오류 발생 → 작업 완료 상태로 복귀


def launch_demo():
    """
    Gradio UI를 실행합니다. 실시간 텍스트 입력 기반 단어별 음성 합성 기능을 제공합니다.
    """
    logging.info("[launch_demo] Starting CosyVoice2 Gradio UI (word-by-word TTS mode)")

    with gr.Blocks() as demo:
        gr.Markdown("## CosyVoice2: Type-and-speak by word (space-delimited)")
        gr.Markdown("단어 입력 후 스페이스를 누르면 해당 단어만 음성으로 합성됩니다. (보조 문장 없이 전체 음성 출력)")

        # 텍스트 입력 + 시드 + 작업 중 상태
        is_processing = gr.State(value=False)

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

        # 출력
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

        # 텍스트 박스 변경 시 -> Debounce TTS 실행
        textbox.change(
            fn=generate_tts_safe,
            inputs=[textbox, seed, is_processing],
            outputs=[audio_output, debug_output, is_processing]
        )

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
