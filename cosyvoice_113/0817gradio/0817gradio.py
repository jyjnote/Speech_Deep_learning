#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import threading
import queue
import time
import numpy as np
import torch
import gradio as gr

# 프로젝트 루트 기준으로 third_party/Matcha-TTS 등이 필요한 경우 경로 추가
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, "third_party/Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging

# -------------------------
# 기본 경로 (CLI로 덮어쓸 수 있음)
# -------------------------
DEFAULT_MODEL_DIR = "/mnt/raid0/jjy/CosyVoice/pretrained_models/CosyVoice2-0.5B"
DEFAULT_PROMPT_WAV = "/mnt/raid0/jjy/CosyVoice/asset/zero_shot_prompt.wav"
DEFAULT_PROMPT_TEXT = "This is a reference voice for zero-shot cloning."

# 전역 상태
cosy = None
prompt_speech_16k = None
MODEL_DIR_SELECTED = DEFAULT_MODEL_DIR
PROMPT_WAV_SELECTED = DEFAULT_PROMPT_WAV

# 세션별 상태 (텍스트 입력 큐, 종료 이벤트)
_sessions = {}
# 입력 코얼레싱/중복 방지용 캐시
_last_sent_text = {}


def _new_session():
    """텍스트 입력을 전달할 큐와 종료 이벤트 생성."""
    text_q = queue.Queue()
    stop_event = threading.Event()
    stop_event.clear()
    return {"q": text_q, "stop": stop_event}


def _text_stream_generator(sess):
    """
    Gradio .input 이벤트로 들어오는 텍스트 '최신 스냅샷'만 제너레이터로 전달.
    - 폴링은 0.5초(비어 있을 때만 대기)
    - 큐를 비워 코얼레싱(가장 최신값 하나만 유지)
    """
    q = sess["q"]
    stop_event = sess["stop"]
    while not stop_event.is_set():
        try:
            # 비었으면 최대 0.5초 대기 (CPU 절약)
            item = q.get(timeout=0.5)
        except queue.Empty:
            continue

        if item is None:  # 종료 신호
            break

        # 🔹 코얼레싱: 큐를 비워서 '가장 최신 스냅샷'만 남김
        latest = item
        while True:
            try:
                nxt = q.get_nowait()
                if nxt is None:
                    latest = None
                    break
                latest = nxt
            except queue.Empty:
                break

        if latest is None:
            break
        yield latest


def start_stream(session_id: str,
                 chunk_tokens=None,
                 interleave_prompt=False,
                 speed=1.0,
                 text_frontend=True):
    """
    스트리밍 합성 시작. Audio 컴포넌트로 (sr, wave) 스트림을 지속적으로 yield.
    (UI에서 고급 옵션을 제거했지만, 내부 기본값 인자를 유지)
    """
    global cosy, prompt_speech_16k

    if session_id not in _sessions:
        _sessions[session_id] = _new_session()
    sess = _sessions[session_id]

    # CosyVoice2의 bi-stream 타이핑 합성 제너레이터 구성
    gen = cosy.inference_zero_shot_typing(
        text_stream=_text_stream_generator(sess),          # <- 입력 스트림 (텍스트 버퍼들이 들어옴, 최신 스냅샷만)
        prompt_text=DEFAULT_PROMPT_TEXT,
        prompt_speech_16k=prompt_speech_16k,
        zero_shot_spk_id="",
        stream=True,
        speed=float(speed),
        text_frontend=bool(text_frontend),
        chunk_tokens=int(chunk_tokens) if chunk_tokens and int(chunk_tokens) > 0 else None,
        interleave_prompt_in_llm=bool(interleave_prompt),
    )

    # 제너레이터가 내는 음성청크를 Audio로 스트리밍
    try:
        for out in gen:
            wav = out["tts_speech"].squeeze(0).cpu().numpy()  # (N,)
            yield (cosy.sample_rate, wav)
    except Exception as e:
        logging.error(f"[start_stream] streaming error: {e}")
    finally:
        # 끝 처리 (세션 유지, stop은 따로)
        pass


def on_text_input(session_id, current_text):
    """
    텍스트박스 .input 이벤트 핸들러.
    - 같은 내용 반복 전송은 무시
    - 큐를 비우고(코얼레싱) 최신 스냅샷 1건만 넣어 폭주 방지
    """
    if session_id not in _sessions:
        _sessions[session_id] = _new_session()
    sess = _sessions[session_id]
    txt = current_text or ""

    # 같은 내용이면 무시
    if _last_sent_text.get(session_id) == txt:
        return
    _last_sent_text[session_id] = txt

    # 큐 비우기(코얼레싱). 중간에 stop 센티널(None)을 만나면 복원
    try:
        while True:
            v = sess["q"].get_nowait()
            if v is None:
                sess["q"].put(None)
    except queue.Empty:
        pass

    # 최신 스냅샷 1건만 넣기
    sess["q"].put(txt)


def stop_stream(session_id):
    """스트리밍 종료: 제너레이터에 종료 신호(None) 전달."""
    if session_id in _sessions:
        _sessions[session_id]["stop"].set()
        _sessions[session_id]["q"].put(None)
    return None  # Audio 초기화(정지)


def clear_textbox():
    return ""


def build_demo():
    with gr.Blocks(title="CosyVoice2 Typing-Stream TTS", css="""
    #audio {min-height: 80px;}
    """) as demo:
        gr.Markdown("## CosyVoice2 실시간 타이핑 TTS (n:m bi-stream)")
        gr.Markdown("모델과 프롬프트는 **서버 시작 시 선 로드됨**. 아래 경로는 정보 표시용입니다.")

        with gr.Row():
            model_dir_info = gr.Textbox(
                label="Model Dir (preloaded)",
                value=MODEL_DIR_SELECTED,
                interactive=False
            )
            prompt_wav_info = gr.Textbox(
                label="Prompt WAV (preloaded, 16kHz)",
                value=PROMPT_WAV_SELECTED,
                interactive=False
            )

        with gr.Row():
            text = gr.Textbox(
                label="Type here (실시간 입력)",
                lines=4,
                placeholder="타이핑하면 바로바로 합성이 시작됩니다. (영문 권장)"
            )

        # Advanced(토큰/스트리밍 제어) 섹션 제거

        audio = gr.Audio(label="Streaming Audio", streaming=True, autoplay=True, elem_id="audio")
        with gr.Row():
            start_btn = gr.Button("▶ Start", variant="primary")
            stop_btn = gr.Button("⏹ Stop", variant="stop")
            clear_btn = gr.Button("🧹 Clear Text")

        # 세션 식별용 (간단히 timestamp 기반)
        session_id = gr.State(value=str(time.time()))

        # Start: 선로드된 모델/프롬프트 기반으로 바로 스트림 시작
        # ⭐ 핵심: 함수가 제너레이터를 '반환(return)'하지 말고, '직접 yield' 해야 함
        def _start(sid):
            # 한글에서 과도한 2-토큰 청크 폭주 방지용: n=8 권장
            yield from start_stream(sid, 8, False, 1.0, True)

        start_btn.click(
            _start,
            inputs=[session_id],
            outputs=[audio],
            queue=True
        )

        # Stop
        stop_btn.click(stop_stream, inputs=[session_id], outputs=[audio])

        # Text 변화 -> 입력 큐에 현재 버퍼 전달 (디바운스 0으로 키 입력마다)
        text.input(on_text_input, inputs=[session_id, text], outputs=[])

        # Clear text
        clear_btn.click(clear_textbox, inputs=[], outputs=[text])

    return demo


def main():
    global cosy, prompt_speech_16k, MODEL_DIR_SELECTED, PROMPT_WAV_SELECTED

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt-wav", type=str, default=DEFAULT_PROMPT_WAV)
    args = parser.parse_args()

    # 1) 모델/프롬프트 **선 로드**
    MODEL_DIR_SELECTED = args.model_dir
    PROMPT_WAV_SELECTED = args.prompt_wav

    logging.info(f"[init] loading model from: {MODEL_DIR_SELECTED}")
    cosy = CosyVoice2(MODEL_DIR_SELECTED, fp16=torch.cuda.is_available())

    logging.info(f"[init] loading prompt wav: {PROMPT_WAV_SELECTED}")
    prompt_speech_16k = load_wav(PROMPT_WAV_SELECTED, 16000)

    # 2) 웹 UI 실행 (선 로드 완료 후)
    demo = build_demo()
    demo.queue(max_size=8, default_concurrency_limit=2)
    demo.launch(server_name=args.server_name, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
