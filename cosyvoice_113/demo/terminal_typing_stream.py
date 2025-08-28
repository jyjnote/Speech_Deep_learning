#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import torch
import torchaudio

# CosyVoice import
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging

sys.path.append('third_party/Matcha-TTS')
# -------------------------
# 터미널(raw) 입력을 실시간으로 읽어 스트림으로 내보내는 제너레이터
# - 일반 키: 그대로 누적 + 즉시 전달
# - Backspace: 로컬 버퍼에서 1글자 삭제 후 ("RESET", 전체텍스트) 전송
# - Enter: 공백으로 처리 (" ")
# - Ctrl-R: 전체 리셋 ("RESET", "")
# - Ctrl-D/Ctrl-C: 종료
# -------------------------

# chunk_tokens를 주는이유 바로 없이 넘기면 실제로 입력으로 넘어가는건 'h','e','l','l','o' hello 가 아닌 h,,,e임
def terminal_text_stream(throttle_sec: float = 0.0):
    """
    throttle_sec>0 으로 두면, 너무 잦은 이벤트를 묶어서 보내도록 튜닝 가능.
    (여기선 실시간 감각을 위해 기본 0.0)
    """
    import select, termios, tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)  # 라인버퍼링 끄고 1글자씩 읽기
        buf = []           # 화면상/모델상 현재 누적 텍스트
        last_emit = 0.0

        print("\n[Terminal-Stream] 실시간 입력 시작")
        print("  - 일반 키: 즉시 합성에 반영")
        print("  - Backspace: 1글자 삭제")
        print("  - Enter: 공백 1개 추가")
        print("  - Ctrl-R: 전체 리셋")
        print("  - Ctrl-D/Ctrl-C: 종료\n")
        print("입력 중...\n", flush=True)

        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if sys.stdin in r:
                ch = sys.stdin.read(1)

                # 종료 키
                if ch in ('\x04', '\x03'):   # Ctrl-D, Ctrl-C
                    print("\n[Terminal-Stream] 종료 신호 수신")
                    return

                # 전체 리셋
                if ch == '\x12':  # Ctrl-R
                    buf.clear()
                    sys.stdout.write("\n[RESET]\n")
                    sys.stdout.flush()
                    yield ("RESET", "")
                    continue

                # Backspace
                if ch == '\x7f':
                    if buf:
                        buf.pop()
                        # 터미널에서 지워보이게
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                        # 모델에도 즉시 반영
                        yield ("RESET", "".join(buf))
                    continue

                # Enter -> 공백
                if ch in ('\r', '\n'):
                    ch = ' '
                    sys.stdout.write(' ')
                    sys.stdout.flush()
                    buf.append(ch)
                    yield ch
                    continue

                # 일반 문자
                # (터미널 echo)
                if ch.isprintable():
                    sys.stdout.write(ch)
                    sys.stdout.flush()
                else:
                    # 기타 제어문자는 무시
                    continue

                buf.append(ch)

                # throttle 없이 즉시 전송
                now = time.time()
                if throttle_sec <= 0 or (now - last_emit) >= throttle_sec:
                    last_emit = now
                    yield ch

            # select timeout: 아무 것도 안 함(keep loop)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    parser = argparse.ArgumentParser(description="CosyVoice2: Terminal typing bi-stream TTS")
    parser.add_argument("--model_dir", type=str, default="pretrained_models/CosyVoice2-0.5B")
    parser.add_argument("--prompt_wav", type=str, default="asset/zero_shot_prompt.wav")
    parser.add_argument("--prompt_text", type=str, default="Hope you can outdo me one day!")
    parser.add_argument("--chunk_tokens", type=int, default=10, help="LLM 청크당 텍스트 토큰수")
    parser.add_argument("--no_interleave_prompt_in_llm", action="store_true",
                        help="LLM에 프롬프트 음성토큰을 섞지 않도록 강제")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--out_prefix", type=str, default="./terminal_stream")
    args = parser.parse_args()

    # 0) 모델 로드
    cosyvoice = CosyVoice2(
        model_dir=args.model_dir,
        load_jit=False, load_trt=False, load_vllm=False, fp16=args.fp16
    )

    # 1) 프롬프트 오디오(16k)
    if not os.path.exists(args.prompt_wav):
        raise FileNotFoundError(f"prompt_wav not found: {args.prompt_wav}")
    prompt_speech_16k = load_wav(args.prompt_wav, 16000)

    # 2) 터미널 입력 제너레이터 생성
    text_stream = terminal_text_stream(throttle_sec=0.0)

    # 3) 스트리밍 합성 시작 (bi-stream)
    gen = cosyvoice.inference_zero_shot_typing(
        text_stream=text_stream,
        prompt_text=args.prompt_text,
        prompt_speech_16k=prompt_speech_16k,
        zero_shot_spk_id="",
        stream=True,
        speed=1.0,
        text_frontend=True,
        chunk_tokens=args.chunk_tokens,
        interleave_prompt_in_llm=(not args.no_interleave_prompt_in_llm),
    )

    sr = cosyvoice.sample_rate
    all_chunks = []
    print(f"\n[Streaming] sample_rate={sr} / out_prefix={args.out_prefix}\n")

    try:
        for i, out in enumerate(gen, 1):
            wav = out["tts_speech"].cpu()  # [1, T]
            out_path = f"{args.out_prefix}_ko_{i:02d}.wav"
            torchaudio.save(out_path, wav, sr)
            all_chunks.append(wav)
            logging.info(f"[saved] {out_path} shape={tuple(wav.shape)}")
    except KeyboardInterrupt:
        print("\n[Streaming] KeyboardInterrupt, finalize...")

    # 4) 최종 합치기
    if all_chunks:
        final_wav = torch.cat(all_chunks, dim=1)
        final_path = f"{args.out_prefix}_final.wav"
        torchaudio.save(final_path, final_wav, sr)
        print(f"\n[Done] Final merged audio saved: {final_path} shape={tuple(final_wav.shape)}")
    else:
        print("\n[Done] No audio chunks were produced.")


if __name__ == "__main__":
    main()

#CUDA_VISIBLE_DEVICES=4 python -u cosyvoice/demo/terminal_typing_stream.py   
# --model_dir pretrained_models/CosyVoice2-0.5B   --prompt_wav asset/zero_shot_prompt.wav   
# --prompt_text "Hope you can outdo me one day!"   --chunk_tokens 10   --out_prefix ./typing_stream

#Only fools rush in But I can't help Falling in love with you

#Shall I stay? Would it be a sin If I can't help Falling in love with you?

#Like a river flows Surely to the sea Darling, so it goes Some things are meant to be

#Take my hand Take my whole life too For I can't help Falling in love with you

#Like a river flows Surely to the sea Darling, so it goes Some things are meant to be

#Take my hand Take my whole life too For I can't help Falling in love with you

#For I can't help Falling in love with you