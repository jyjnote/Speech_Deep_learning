from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from matcha.hifigan.models import feature_loss, generator_loss, discriminator_loss
from cosyvoice.utils.losses import tpr_loss, mel_loss


class HiFiGan(nn.Module):
    def __init__(self, generator, discriminator, mel_spec_transform,
                 multi_mel_spectral_recon_loss_weight=45, feat_match_loss_weight=2.0,
                 tpr_loss_weight=1.0, tpr_loss_tau=0.04):
        super(HiFiGan, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.mel_spec_transform = mel_spec_transform
        self.multi_mel_spectral_recon_loss_weight = multi_mel_spectral_recon_loss_weight
        self.feat_match_loss_weight = feat_match_loss_weight
        self.tpr_loss_weight = tpr_loss_weight
        self.tpr_loss_tau = tpr_loss_tau

        # ✅ weight_norm이 제거된 checkpoint와 호환되도록 시도
        removed_count = 0
        for name, module in self.generator.named_modules():
            try:
                remove_weight_norm(module)
                removed_count += 1
            except Exception:
                pass
        print(f"[INFO] Removed weight norm from {removed_count} modules in generator.")

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if batch['turn'] == 'generator':
            return self.forward_generator(batch, device)
        else:
            return self.forward_discriminator(batch, device)

    def forward_generator(self, batch, device):
        real_speech = batch['speech'].to(device)
        pitch_feat = batch['pitch_feat'].to(device)

        # 1. calculate generator outputs
        generated_speech, generated_f0 = self.generator(batch, device)

        # 🔧 pitch_feat 길이에 맞게 generated_f0 보정
        f0_len = pitch_feat.shape[1]
        gen_f0_len = generated_f0.shape[1]
        if gen_f0_len > f0_len:
            #print(f"[forward_generator] Trimming generated_f0 from {gen_f0_len} to {f0_len}")
            generated_f0 = generated_f0[:, :f0_len]
        elif gen_f0_len < f0_len:
            pad_len = f0_len - gen_f0_len
            #print(f"[forward_generator] Padding generated_f0 from {gen_f0_len} to {f0_len} (pad_len={pad_len})")
            generated_f0 = F.pad(generated_f0, (0, pad_len), mode='constant', value=0.0)
        else:
            #print(f"[forward_generator] No trimming or padding needed for f0")
            pass

        # 🔍 Debug: 출력 크기 확인
        #print(f"[forward_generator] real_speech.shape: {real_speech.shape}")
        #print(f"[forward_generator] generated_speech.shape: {generated_speech.shape}")

        # 🔧 generated_speech 길이 real_speech에 맞추기
        target_len = real_speech.shape[-1]
        gen_len = generated_speech.shape[-1]

        if gen_len > target_len:
            #print(f"[forward_generator] Trimming generated_speech from {gen_len} to {target_len}")
            generated_speech = generated_speech[:, :target_len]
        elif gen_len < target_len:
            pad_len = target_len - gen_len
            #print(f"[forward_generator] Padding generated_speech from {gen_len} to {target_len} (pad_len={pad_len})")
            generated_speech = F.pad(generated_speech, (0, pad_len), mode='constant', value=0.0)
        else:
            #print(f"[forward_generator] No trimming or padding needed (lengths match)")
            pass
        # 2. calculate discriminator outputs
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.discriminator(real_speech, generated_speech)

        # 3. calculate generator losses
        loss_gen, _ = generator_loss(y_d_gs)
        loss_fm = feature_loss(fmap_rs, fmap_gs)
        loss_mel = mel_loss(real_speech, generated_speech, self.mel_spec_transform)

        if self.tpr_loss_weight != 0:
            loss_tpr = tpr_loss(y_d_gs, y_d_rs, self.tpr_loss_tau)
        else:
            loss_tpr = torch.zeros(1).to(device)

        loss_f0 = F.l1_loss(generated_f0, pitch_feat)

        loss = loss_gen + \
            self.feat_match_loss_weight * loss_fm + \
            self.multi_mel_spectral_recon_loss_weight * loss_mel + \
            self.tpr_loss_weight * loss_tpr + \
            loss_f0

        # 🔍 Debug: 손실 출력
        print(f"[forward_generator] Losses: total={loss.item():.4f}, gen={loss_gen.item():.4f}, "
              f"fm={loss_fm.item():.4f}, mel={loss_mel.item():.4f}, tpr={loss_tpr.item():.4f}, f0={loss_f0.item():.4f}")

        return {
            'loss': loss,
            'loss_gen': loss_gen,
            'loss_fm': loss_fm,
            'loss_mel': loss_mel,
            'loss_tpr': loss_tpr,
            'loss_f0': loss_f0
        }


    def forward_discriminator(self, batch, device):
        real_speech = batch['speech'].to(device)

        # 1. calculate generator outputs
        with torch.no_grad():
            generated_speech, generated_f0 = self.generator(batch, device)

            # ✅ 길이 자동 정렬
            real_len = real_speech.shape[-1]
            gen_len = generated_speech.shape[-1]

            if gen_len > real_len:
                generated_speech = generated_speech[..., :real_len]
                print(f"[Discriminator] Cropped generated_speech from {gen_len} to {real_len}")
            elif gen_len < real_len:
                pad_len = real_len - gen_len
                generated_speech = F.pad(generated_speech, (0, pad_len))
                print(f"[Discriminator] Padded generated_speech from {gen_len} to {real_len}")

        # 2. calculate discriminator outputs
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.discriminator(real_speech, generated_speech.detach())

        # 3. calculate discriminator losses
        loss_disc, _, _ = discriminator_loss(y_d_rs, y_d_gs)

        if self.tpr_loss_weight != 0:
            loss_tpr = tpr_loss(y_d_rs, y_d_gs, self.tpr_loss_tau)
        else:
            loss_tpr = torch.zeros(1).to(device)

        loss = loss_disc + self.tpr_loss_weight * loss_tpr
        return {'loss': loss, 'loss_disc': loss_disc, 'loss_tpr': loss_tpr}