#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import io
import base64
import torch
import torchaudio
from flask import Flask, Response, render_template_string
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging

sys.path.append('third_party/Matcha-TTS')

app = Flask(__name__)

print("서버 시작: CosyVoice2 모델 로드...")
cosyvoice = CosyVoice2(
    model_dir="pretrained_models/CosyVoice2-0.5B",
    load_jit=False, load_trt=False, load_vllm=False, fp16=False
)
prompt_speech_16k = load_wav("asset/zero_shot_prompt.wav", 16000)
print("모델 로드 완료.")

# --- 터미널 입력 제너레이터 ---
def terminal_text_stream(throttle_sec: float = 0.0, idle_flush_sec: float = 2.0):
    """
    터미널에서 글자를 실시간으로 읽어 CosyVoice로 흘려보내는 제너레이터.
    - throttle_sec: 토큰 전송 최소 간격
    - idle_flush_sec: 입력이 idle 상태로 유지되는 최대 시간(초)
                      넘으면 지금까지 buffer 전체를 RESET 신호로 flush
    """
    import select, termios, tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        buf = []
        last_emit = 0.0
        last_input_time = time.time()

        print("\n[Terminal-HTTP] 입력 시작 (Ctrl-D/Ctrl-C 종료)")
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            now = time.time()

            # --- idle flush 체크 ---
            #if (now - last_input_time) >= idle_flush_sec and buf:
            #    full_text = "".join(buf)
            #    print(f"\n[Terminal-HTTP] idle {idle_flush_sec}s → flush '{full_text}'")
            #    yield ("RESET", full_text)
            #    buf.clear()
            #    last_input_time = now

            if sys.stdin in r:
                ch = sys.stdin.read(1)
                last_input_time = time.time()  # 마지막 입력시각 갱신

                if ch in ('\x04', '\x03'):   # Ctrl-D / Ctrl-C
                    print("[Terminal-HTTP] 종료")
                    return

                if ch in ('\r', '\n'):       # Enter -> 공백
                    ch = ' '
                    sys.stdout.write(' ')
                    sys.stdout.flush()
                    buf.append(ch)
                    yield ch
                    continue

                if ch.isprintable():        # 일반 문자
                    sys.stdout.write(ch)
                    sys.stdout.flush()
                    buf.append(ch)
                    if throttle_sec <= 0 or (time.time() - last_emit) >= throttle_sec:
                        last_emit = time.time()
                        yield ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# --- 오디오 이벤트 생성 ---
def generate_audio_events():
    sr = cosyvoice.sample_rate
    gen = cosyvoice.inference_zero_shot_typing(
        text_stream=terminal_text_stream(),
        prompt_text="Hope you can outdo me one day!",
        prompt_speech_16k=prompt_speech_16k,
        stream=True, speed=1.0, text_frontend=True,
        chunk_tokens=20, interleave_prompt_in_llm=False,
    )

    for i, out in enumerate(gen, 1):
        wav = out["tts_speech"].cpu()
        logging.info(f"[청크] {i} shape={tuple(wav.shape)}")

        buffer = io.BytesIO()
        torchaudio.save(buffer, wav, sr, format="wav")
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("utf-8")

        yield f"data: {encoded}\n\n"

    yield "data: END_OF_STREAM\n\n"


@app.route("/audio_events")
def audio_events():
    return Response(generate_audio_events(), mimetype="text/event-stream")


@app.route("/")
def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>Terminal Typing Stream</title></head>
    <body>
      <h1>CosyVoice2 Terminal Typing Stream</h1>
      <p id="status">연결 중...</p>
      <audio id="player" controls></audio>
      <script>
        const status=document.getElementById('status');
        const player=document.getElementById('player');
        const queue=[]; let playing=false;

        const es=new EventSource("/audio_events");
        es.onmessage=function(e){
          if(e.data.startsWith("END_OF_STREAM")){
            status.textContent="스트리밍 종료";
            es.close(); return;
          }
          const url="data:audio/wav;base64,"+e.data;
          queue.push(url);
          status.textContent="청크 "+queue.length+" 수신";
          if(!playing) playNext();
        };
        function playNext(){
          if(queue.length>0){
            playing=true;
            player.src=queue.shift();
            player.play().catch(()=>{playing=false;});
          } else {
            playing=false;
          }
        }
        player.onended=function(){playNext();};
      </script>
    </body>
    </html>
    """
    return render_template_string(html)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
