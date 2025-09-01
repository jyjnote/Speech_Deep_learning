#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import io
import re
import time
import json
import base64
import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import torch
import torchaudio
from flask import Flask, request, Response, render_template_string, jsonify

# ==============================
# 0) App & Model Init (경로 설정 포함)
# ==============================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # .../CosyVoice/demo
PROJ_DIR = os.path.dirname(THIS_DIR)                          # .../CosyVoice

# 패키지 경로 등록
sys.path.append(PROJ_DIR)
sys.path.append(os.path.join(PROJ_DIR, "third_party", "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging

app = Flask(__name__)

MODEL_DIR = os.path.join(PROJ_DIR, "pretrained_models", "CosyVoice2-0.5B")
PROMPT_WAV = os.path.join(PROJ_DIR, "asset", "zero_shot_prompt1.wav")

if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(f"{MODEL_DIR} does not exist! (expected CosyVoice2-0.5B)")
if not os.path.isfile(PROMPT_WAV):
    raise FileNotFoundError(f"{PROMPT_WAV} not found!")

print("서버 시작: CosyVoice2 로드…")
cosyvoice = CosyVoice2(model_dir=MODEL_DIR, fp16=False)
prompt_speech_16k = load_wav(PROMPT_WAV, 16000)
print("모델 로드 완료.")

SAMPLE_RATE = cosyvoice.sample_rate
PUNCT = re.compile(r"[\.!\?…。\！？]")  # 문장 종료 구두점

# ==============================
# 1) 세션 관리
# ==============================
@dataclass
class Session:
    sid: str
    text: str = ""                     # 사용자의 전체 텍스트
    last_flush_idx: int = 0            # 마지막 플러시 인덱스
    last_input_ts: float = field(default_factory=time.time)
    tts_q: queue.Queue = field(default_factory=queue.Queue)      # 합성 대기 문장 큐
    sse_q: queue.Queue = field(default_factory=queue.Queue)      # 브라우저로 보낼 오디오/상태 큐
    worker_thread: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)

SESSIONS: Dict[str, Session] = {}
SESS_LOCK = threading.Lock()

IDLE_FLUSH_SEC = 1.5   # 타자 멈춤 시 강제 플러시
KEEPALIVE_SEC   = 15.0 # SSE keep-alive 주기

def get_or_create_session(sid: str) -> Session:
    with SESS_LOCK:
        sess = SESSIONS.get(sid)
        if sess is None:
            sess = Session(sid=sid)
            SESSIONS[sid] = sess
            t = threading.Thread(target=tts_worker, args=(sess,), daemon=True)
            t.start()
            sess.worker_thread = t
    return sess

# ==============================
# 2) 문장 추출 & 큐잉
# ==============================
def enqueue_flushable_sentences(sess: Session, force: bool = False):
    """
    sess.text 의 last_flush_idx 이후에서
    - 구두점으로 끝난 문장들을 찾아 tts_q 에 enqueue
    - force=True면 남은 잔여도 강제 플러시
    """
    new_segment = sess.text[sess.last_flush_idx:]
    if not new_segment:
        return

    consumed = 0
    sentences: List[str] = []

    # 구두점 기준 문장 수집
    # 예: "안녕. 반가워!" -> "안녕.", "반가워!"
    for m in re.finditer(r"[^\.!\?…。\！？]*[\.!\?…。\！？]", new_segment):
        end = m.end()
        chunk = new_segment[:end].strip()
        if chunk:
            sentences.append(chunk)
        new_segment = new_segment[end:]
        consumed += end

    sess.last_flush_idx += consumed

    # force이면 잔여도 문장으로 처리
    if force:
        rest = new_segment.strip()
        if rest:
            sentences.append(rest)
            sess.last_flush_idx = len(sess.text)

    for s in sentences:
        logging.debug(f"[{sess.sid}] enqueue sentence: {s[:80]}{'...' if len(s)>80 else ''}")
        sess.tts_q.put(s)

# ==============================
# 3) TTS 워커 (문장 → WAV → SSE)
# ==============================
def synth_sentence_to_wav_bytes(sentence: str) -> bytes:
    """
    문장 하나를 비-스트리밍(stream=False)으로 합성하여 WAV 바이너리 반환.
    front-end가 내부적으로 chunk를 나누면 이어 붙여서 하나로 보냄.
    """
    chunks = cosyvoice.frontend.text_normalize(sentence, split=True, text_frontend=True)
    wav_parts = []

    for nc in chunks:
        # 보이스 클로닝(Zero-shot) + LLM interleave off
        mi = cosyvoice.frontend.frontend_zero_shot(
            nc, "<|endofprompt|>", prompt_speech_16k, cosyvoice.sample_rate, zero_shot_spk_id=""
        )
        # LLM에 프롬프트 음성토큰 섞지 않기(지연/전이 제어)
        mi["llm_prompt_speech_token"] = torch.zeros(1, 0, dtype=torch.int32, device=cosyvoice.model.device)

        for out in cosyvoice.model.tts(**mi, stream=False, speed=1.0):
            wav_parts.append(out["tts_speech"].cpu())

    if not wav_parts:
        return b""

    wav_cat = torch.cat(wav_parts, dim=1)  # [1, T]
    buf = io.BytesIO()
    torchaudio.save(buf, wav_cat, SAMPLE_RATE, format="wav")
    buf.seek(0)
    return buf.read()

def tts_worker(sess: Session):
    last_keepalive = time.time()

    while not sess.stop_event.is_set():
        now = time.time()

        # 서버측 Idle flush
        if (now - sess.last_input_ts) >= IDLE_FLUSH_SEC and sess.last_flush_idx < len(sess.text):
            enqueue_flushable_sentences(sess, force=True)

        # 합성 처리
        try:
            sentence = sess.tts_q.get(timeout=0.1)
        except queue.Empty:
            sentence = None

        if sentence:
            try:
                wav_bytes = synth_sentence_to_wav_bytes(sentence)
                if wav_bytes:
                    b64 = base64.b64encode(wav_bytes).decode("utf-8")
                    sess.sse_q.put(json.dumps({"type": "audio", "b64wav": b64}))
                else:
                    sess.sse_q.put(json.dumps({"type": "log", "msg": "empty audio"}))
            except Exception as e:
                sess.sse_q.put(json.dumps({"type": "log", "msg": f"TTS error: {e}"}))

        # keepalive
        if (time.time() - last_keepalive) >= KEEPALIVE_SEC:
            sess.sse_q.put(json.dumps({"type": "ping"}))
            last_keepalive = time.time()

    sess.sse_q.put(json.dumps({"type": "end"}))

# ==============================
# 4) HTTP Routes
# ==============================
@app.route("/")
def index():
    html = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>Streaming Text → TTS (Sentence Flush)</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, 'Noto Sans KR', sans-serif; line-height:1.4; padding:24px; }
    textarea { width: 100%; height: 160px; font-size: 16px; }
    .log { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background:#f6f7f9; padding:12px; border-radius:8px; }
    .row { display:flex; gap:12px; align-items:center; }
    .small { font-size:12px; color:#666; }
  </style>
</head>
<body>
  <h1>Streaming Text → TTS (문장 플러시)</h1>
  <div class="row">
    <textarea id="ta" placeholder="여기에 계속 타이핑 해보세요. 구두점(. ? ! … 등) 또는 잠시 멈추면 문장 단위로 합성됩니다."></textarea>
  </div>
  <p class="small">문장 완료 시마다 음성이 자동 재생됩니다. (SSE, 문장 단위 WAV)</p>
  <audio id="player" controls></audio>
  <div class="log" id="log"></div>

<script>
(function() {
  const sid = crypto.randomUUID();
  const ta = document.getElementById('ta');
  const log = document.getElementById('log');
  const player = document.getElementById('player');

  // ---- 오디오 재생 큐 ----
  const q = [];
  let playing = false;
  function enqueueAndPlay(b64) {
    // Blob 사용 (data URL보다 안정적)
    const byteChars = atob(b64);
    const byteNums = new Array(byteChars.length);
    for (let i=0; i<byteChars.length; i++) byteNums[i] = byteChars.charCodeAt(i);
    const blob = new Blob([new Uint8Array(byteNums)], {type: 'audio/wav'});
    const url = URL.createObjectURL(blob);
    q.push(url);
    if (!playing) playNext();
  }
  function playNext() {
    if (q.length === 0) { playing = false; return; }
    playing = true;
    const url = q.shift();
    player.src = url;
    player.play().catch(err => {
      log.textContent += "\\n[AUDIO] play error: " + err;
      playing = false;
    });
  }
  player.onended = () => playNext();

  // ---- SSE 연결 (오디오 수신) ----
  const es = new EventSource("/sse_audio?sid=" + encodeURIComponent(sid));
  es.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.type === 'audio' && msg.b64wav) {
        enqueueAndPlay(msg.b64wav);
      } else if (msg.type === 'log') {
        log.textContent += "\\n" + msg.msg;
      } else if (msg.type === 'ping') {
        // keepalive
      } else if (msg.type === 'end') {
        log.textContent += "\\n[SSE] end";
        es.close();
      }
    } catch (err) {
      log.textContent += "\\n[SSE] parse error: " + err;
    }
  };

  // ---- 입력 이벤트: 디바운스 전송 + 구두점 즉시 플러시 ----
  const PUNCT = /[\\.!\\?…。！？]/;
  let sendTimer = null;
  function sendText(force=false) {
    const text = ta.value;
    fetch('/type', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ sid, text, force })
    }).catch(()=>{});
  }

  ta.addEventListener('input', () => {
    const val = ta.value;
    // 구두점이 막 입력되면 즉시 플러시
    if (val.length > 0 && PUNCT.test(val[val.length-1])) {
      sendText(true);
      return;
    }
    // 그 외에는 디바운스(150ms)
    if (sendTimer) clearTimeout(sendTimer);
    sendTimer = setTimeout(() => sendText(false), 150);
  });

  ta.addEventListener('change', () => sendText(false));
})();
</script>
</body>
</html>
    """
    return render_template_string(html)

@app.route("/type", methods=["POST"])
def type_event():
    """
    프론트에서 현재 전체 텍스트를 계속 보내줌.
    - force=True 이거나, 서버 idle 감지 시 문장 플러시
    """
    data = request.get_json(force=True)
    sid = data.get("sid") or str(uuid.uuid4())
    text = data.get("text", "")
    force = bool(data.get("force", False))

    sess = get_or_create_session(sid)
    sess.text = text
    sess.last_input_ts = time.time()

    enqueue_flushable_sentences(sess, force=force)
    return jsonify({"ok": True})

@app.route("/sse_audio")
def sse_audio():
    """
    문장 단위로 생성된 오디오를 SSE로 전송.
    """
    sid = request.args.get("sid") or str(uuid.uuid4())
    sess = get_or_create_session(sid)

    def event_stream():
        last_ping = time.time()
        while not sess.stop_event.is_set():
            try:
                msg = sess.sse_q.get(timeout=0.5)
                yield f"data: {msg}\n\n"
            except queue.Empty:
                if (time.time() - last_ping) >= KEEPALIVE_SEC:
                    yield 'data: {"type":"ping"}\n\n'
                    last_ping = time.time()

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Content-Type": "text/event-stream",
        "Connection": "keep-alive",
    }
    return Response(event_stream(), headers=headers)

# ==============================
# 5) Run
# ==============================
if __name__ == "__main__":
    # 예: CUDA_VISIBLE_DEVICES=0 python app_stream_tts.py
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
