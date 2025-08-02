import torch
import re

# 원본 Torch 2.3 모델 (parametrizations 포함)
src_path = "pretrained_models/CosyVoice2-0.5B/hift.pt"
# 변환된 Torch 1.13용 모델 저장 경로
dst_path = "pretrained_models/CosyVoice2-0.5B/hift_compatible_113.pt"

print(f"Loading: {src_path}")
state_dict = torch.load(src_path, map_location='cpu')
converted = {}

for k, v in state_dict.items():
    # ✅ 변환 대상 예: conv.parametrizations.weight.original0 → conv.weight_g
    match_g = re.match(r'^(.*)\.parametrizations\.weight\.original0$', k)
    match_v = re.match(r'^(.*)\.parametrizations\.weight\.original1$', k)
    
    if match_g:
        base = match_g.group(1)
        new_k = f"{base}.weight_g"
        converted[new_k] = v
    elif match_v:
        base = match_v.group(1)
        new_k = f"{base}.weight_v"
        converted[new_k] = v
    else:
        # 그대로 복사 (bias 등)
        converted[k] = v

# 저장
torch.save(converted, dst_path)
print(f"✅ Converted model saved to: {dst_path}")
