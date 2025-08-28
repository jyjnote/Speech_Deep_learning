import time
import torch
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav   # 프로젝트 내 로더 (없으면 librosa.load로 대체)

import sys
sys.path.append('third_party/Matcha-TTS')
# 0) 모델 로드
cosyvoice = CosyVoice2(
    model_dir='pretrained_models/CosyVoice2-0.5B',
    load_jit=False, load_trt=False, load_vllm=False, fp16=False
)

# 1) 프롬프트 오디오
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

# 2) "사용자 타이핑"을 흉내내는 제너레이터(영문)
# 해가 지기 전에 가려했지 너와 내가 있던 그 언덕 풍경 속에 아주 키 작은 그 마음으로 세상을 꿈꾸고 그리며 말했던 곳 이제 여행을 떠나야하는 소중한 내 친구여 때론 다투기도 많이 했지 서로 알 수 없는 오해의 조각들로 하지만 멋쩍은 미소만으로 너는 내가 되고 나도 네가 될 수 있었던 수 많은 기억들 내가 항상 여기 서 있을게 걷다가 지친 네가 나를 볼 수 있게 저기 저 별 위에 그릴꺼야 내가 널 사랑하는 마음 볼 수 있게 너는 내가 되고 나도 네가 될 수 있었던 수 많은 기억들 내가 항상 여기 서 있을게 걷다가 지친 네가 나를 볼 수 있게 저기 저 별 위에 그릴꺼야 내가 널 사랑하는 마음 볼 수 있게 내가 항상 여기 서 있을게 걷다가 지친 네가 나를 볼 수 있게 저기 저 별 위에 그릴꺼야 내가 널 사랑하는 마음 볼 수 있게

# # 문자수 475, 공백 포함 591
# def fake_typing():
#     text_pieces = [
#         "We hold these truths, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness. "\
#         "That to secure these rights, Governments are instituted among Men, deriving their just powers from the consent of the governed, "\
#         "That whenever any Form of Government becomes destructive of these ends, it is the Right of the People to alter or to abolish it, and to institute new Government, "\
#         "laying its foundation on such principles and organizing its powers in such form, as to them shall seem most likely to effect their Safety and Happiness."]
#     for piece in text_pieces:
#         yield piece
#         # 각 구절 사이에 1.2초의 지연 시간을 줍니다.
#         # 현재 이건 직접 타이핑이 아니라 그냥 딜레이를 줘서 사용자가 이렇게 넘겨줬다 가정하고 해본 스크립트
#         #time.sleep(1.2)

# 글자수 244, # 공백 포함 370        
def fake_typing(): 
    text_pieces = [
        "우리는 끊임없이 변하는 세상 속에서 때로는 희망을 잃기도 하고 때로는 절망 속에 주저앉기도 하지만,\
        그럼에도 불구하고 마음 깊은 곳에서 꺼지지 않는 불씨와 같은 용기가 우리를 다시 일어서게 만들며, \
        사랑하는 사람들과 함께 나눈 작은 미소와 따뜻한 손길이 인생의 무게를 견뎌낼 힘을 주고, \
        시간이 흐르면서 쌓여가는 수많은 기억들은 결국 우리의 삶을 풍요롭게 하는 자양분이 되어, \
        지금 이 순간을 살아가는 우리 모두가 서로의 존재를 통해 배우고 성장하며, \
        끝내는 서로의 발자취 속에서 세상이 조금 더 따뜻하고 아름다운 곳으로 나아갈 수 있음을 증명하게 된다."
    ]
    for piece in text_pieces:
        yield piece
        #time.sleep(1.2)

# 3) 스트리밍 합성 호출
gen = cosyvoice.inference_zero_shot_typing(
    text_stream=fake_typing(),
    prompt_text="Hope you can outdo me one day!",
    prompt_speech_16k=prompt_speech_16k,
    zero_shot_spk_id="",
    stream=True,
    speed=1.0,
    text_frontend=True,
    chunk_tokens=50, # 사용자의 Raw 텍스트를 토크나이저(BPE)로 변환했을 때의 토큰 ID 개수 
    interleave_prompt_in_llm=False
)
# chunk_tokens 40 token_hop_len 25 28개 37초  ,, token_hop_len 5 139개 27초
# chunk_tokens 20 token_hop_len 25 28개 30초  ,, token_hop_len 5 197개 39초

# 4) 출력 조각 저장 + 메모리에 모아두기
sr = cosyvoice.sample_rate
all_chunks = []

for i, out in enumerate(gen, 1):
    wav = out["tts_speech"].cpu()  # [1, T] 모양
    torchaudio.save(f"./typing_stream_{i:02d}.wav", wav, sr)  # 청크별 저장
    all_chunks.append(wav)  # 최종 합치기 위해 보관
    print("saved:", i, wav.shape)

# 5) 최종 오디오 합치기
if all_chunks:
    final_wav = torch.cat(all_chunks, dim=1)  # 시간축으로 이어붙임
    torchaudio.save("./typing_stream_final_output_EN.wav", final_wav, sr)
    print("Final merged audio saved:", final_wav.shape)
