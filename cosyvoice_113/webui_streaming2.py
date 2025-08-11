import os
import sys
import argparse
import gradio as gr
import numpy as np
import random
import time
from queue import Queue

# CosyVoice 경로
sys.path.append("/mnt/raid0/jjy/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging

# ====== 설정 ======
PROMPT_SR = 16000
PROMPT_PATH = "/mnt/raid0/jjy/CosyVoice/asset/zero_shot_prompt.wav"
GAP_SILENCE_MS = 0                 # 단어 사이 무음 (원하면 >0)
PRIME_SILENCE_SAMPLES = 32         # 스트림 프라임용 아주 짧은 무음(양쪽에 1회씩)

# ====== 전역 ======
cosyvoice = None
prompt_speech_16k = None

pending_words = Queue()     # FIFO 대기열
is_streaming = False        # 현재 스트리밍 중 여부
last_committed_idx = 0      # 이미 큐에 넣은(완성된) 단어 개수 포인터


# ---------- 단어 파싱/큐잉 ----------
def _parse_completed_words(text: str):
    """
    끝이 공백이면 모든 단어 완성, 아니면 마지막 단어는 미완성으로 간주.
    return: (words(list), completed_count:int)
    """
    text = (text or "")
    words = text.strip().split()
    if not words:
        return [], 0
    if text.endswith(" "):
        return words, len(words)
    else:
        return words, max(len(words) - 1, 0)


def _enqueue_new_completed_words(text: str):
    """
    마지막 커밋 인덱스 이후의 '새로 완성된' 단어만 큐에 넣는다.
    중간 수정으로 단어 수가 줄었으면 포인터 되돌림.
    """
    global last_committed_idx
    words, completed_count = _parse_completed_words(text)

    if completed_count < last_committed_idx:
        last_committed_idx = completed_count
        return []

    new_words = words[last_committed_idx:completed_count]
    for w in new_words:
        pending_words.put(w)
    last_committed_idx = completed_count
    return new_words


# ---------- 오디오 유틸 ----------
def _silence(sr: int, ms: int):
    if ms <= 0:
        return np.zeros(0, dtype=np.float32)
    n = int(sr * ms / 1000.0)
    return np.zeros(n, dtype=np.float32)


# ---------- Cosy 스트리밍 ----------
def _cosy_stream_text(text: str):
    """
    CosyVoice2(stream=True) 제너레이터에서 (sr, chunk, out) 스트리밍.
    """
    gen = cosyvoice.inference_cross_lingual(
        tts_text=text,
        prompt_speech_16k=prompt_speech_16k,
        zero_shot_spk_id="",
        stream=True
    )
    sr = cosyvoice.sample_rate
    for out in gen:
        chunk = out["tts_speech"].numpy().flatten().astype(np.float32, copy=False)
        yield sr, chunk, out


# ---------- 스트리밍 루프 (A/B 스왑) ----------
def _stream_loop(seed: int, active_player: int):
    """
    큐가 빌 때까지 단어를 순서대로 합성.
    - 현재 active_player(0→A, 1→B)로 스트리밍
    - 단어 하나 끝날 때마다 플레이어를 토글하여 이전 재생 보장
    매 yield: (audio_a_out, audio_b_out, debug_text, active_player_out)
    """
    sr = cosyvoice.sample_rate
    player = active_player
    debug_lines = []

    while not pending_words.empty():
        word = pending_words.get()
        start_t = time.time()
        debug_lines.append(f"[stream] ▶ '{word}' → {'A' if player==0 else 'B'}")

        # 합성 스트리밍
        for sr_i, chunk, out in _cosy_stream_text(word):
            dbg = "\n".join(debug_lines[-20:])
            if player == 0:
                # A로만 청크 전송, B는 변경 없음
                yield (sr_i, chunk), gr.update(), dbg, player
            else:
                yield gr.update(), (sr_i, chunk), dbg, player

        # 단어 사이 선택적 무음
        gap = _silence(sr, GAP_SILENCE_MS)
        if len(gap) > 0:
            if player == 0:
                yield (sr, gap), gr.update(), "\n".join(debug_lines[-20:]), player
            else:
                yield gr.update(), (sr, gap), "\n".join(debug_lines[-20:]), player

        debug_lines.append(f"[stream] ■ '{word}' done in {time.time()-start_t:.2f}s")

        # 다음 단어는 플레이어 토글 → 이전 재생 유지
        player = 1 - player
        yield gr.update(), gr.update(), "\n".join(debug_lines[-10:]), player

    # 큐 종료
    yield gr.update(), gr.update(), "[stream] queue empty", player


# ---------- 콜백 ----------
def on_text_update(current_text: str, seed: int, active_player: int):
    """
    텍스트 변경 시:
    - 새로 완성된 단어를 큐에 넣고
    - 현재 스트리밍 중이면 오디오는 건드리지 않으며 상태만 갱신
    - 스트리밍 중이 아니면 큐를 소비하면서 A/B 스왑으로 스트리밍
    """
    global is_streaming

    added = _enqueue_new_completed_words(current_text)

    if is_streaming:
        msg = "[update] streaming…"
        if added:
            msg += f" queued: {added}"
        yield gr.update(), gr.update(), msg, active_player
        return

    if pending_words.empty():
        yield gr.update(), gr.update(), "[update] idle (no completed word)", active_player
        return

    is_streaming = True
    try:
        # === 프라임: 양쪽 오디오 스트림을 먼저 생성(짧은 무음 1회씩) ===
        sr = cosyvoice.sample_rate
        tiny = np.zeros(PRIME_SILENCE_SAMPLES, dtype=np.float32)
        yield (sr, tiny), (sr, tiny), "[prime] start streams", active_player
        # ========================================================

        for a_out, b_out, dbg, state_out in _stream_loop(seed, active_player):
            yield a_out, b_out, dbg, state_out
    finally:
        is_streaming = False


# ---------- 앱 ----------
def launch_demo(port: int, model_dir: str):
    logging.info("[launch_demo] CosyVoice2 queued streaming with A/B audio swap (prime fix)")

    with gr.Blocks() as demo:
        gr.Markdown("## CosyVoice2 — No-Skip Streaming via A/B Audio Swap (with prime)")

        with gr.Row():
            textbox = gr.Textbox(
                label="Type (press SPACE to commit each word)",
                lines=2,
                placeholder="예: 주형이␣ 생일␣ 축하해␣",
                interactive=True
            )
            seed = gr.Number(value=random.randint(1, 999999), label="Seed", precision=0)

        with gr.Row():
            audio_a = gr.Audio(label="Audio A", autoplay=True, streaming=True, type="numpy", format="wav")
            audio_b = gr.Audio(label="Audio B", autoplay=True, streaming=True, type="numpy", format="wav")

        debug_output = gr.Textbox(label="Debug", lines=10, interactive=False)

        active_player = gr.State(0)  # 0→A, 1→B

        textbox.change(
            fn=on_text_update,
            inputs=[textbox, seed, active_player],
            outputs=[audio_a, audio_b, debug_output, active_player]
        )

    demo.queue(max_size=64, default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=port)


# ---------- 엔트리 ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument("--model_dir", type=str, default="pretrained_models/CosyVoice2-0.5B")
    args = parser.parse_args()

    logging.info(f"[main] Loading CosyVoice2 from {args.model_dir}")
    cosyvoice = CosyVoice2(model_dir=args.model_dir, load_jit=False, load_trt=False, load_vllm=False)
    logging.info("[main] CosyVoice2 loaded")

    # 프롬프트 오디오 캐시
    prompt_speech_16k = load_wav(PROMPT_PATH, target_sr=PROMPT_SR)

    launch_demo(port=args.port, model_dir=args.model_dir)
