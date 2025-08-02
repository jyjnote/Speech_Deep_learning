# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from contextlib import nullcontext
import os

import torch
import torch.distributed as dist

from cosyvoice.utils.train_utils import update_parameter_and_lr, log_per_step, log_per_save, batch_forward, batch_backward, save_model, cosyvoice_join


class Executor:

    def __init__(self, gan: bool = False, ref_model: torch.nn.Module = None, dpo_loss: torch.nn.Module = None):
        self.gan = gan
        self.ref_model = ref_model
        self.dpo_loss = dpo_loss
        self.step = 0
        self.epoch = 0
        self.rank = int(os.environ.get('RANK', 0))
        self.device = torch.device('cuda:{}'.format(self.rank))

    def train_one_epoc(self, model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join, ref_model=None):
        ''' Train one epoch
        '''
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
        logging.info('using accumulate grad, new batch size is {} times'
                    ' larger than before'.format(info_dict['accum_grad']))

        model.train()
        if self.ref_model is not None:
            self.ref_model.eval()

        model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx

                if cosyvoice_join(group_join, info_dict):
                    break

                if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                else:
                    context = nullcontext

                with context():
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict,
                                            ref_model=self.ref_model, dpo_loss=self.dpo_loss)
                    info_dict = batch_backward(model, scaler, info_dict)

                # 🔧 디버깅 출력 (rank 0만 출력)
                if dist.get_rank() == 0:
                    print(f"[DEBUG] Epoch {self.epoch}, Step {self.step}, Batch {batch_idx}")
                    if 'loss_dict' in info_dict:
                        print(f"[DEBUG] Losses: {info_dict['loss_dict']}")

                info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
                log_per_step(writer, info_dict)

                if info_dict['save_per_step'] > 0 and (self.step + 1) % info_dict['save_per_step'] == 0 and \
                (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    dist.barrier()
                    self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                    model.train()

                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1

        dist.barrier()
        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)


    def train_one_epoc_gan(self, model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                        writer, info_dict, scaler, group_join):
        ''' Train one epoch '''
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
        logging.info('using accumulate grad, new batch size is {} times'
                    ' larger than before'.format(info_dict['accum_grad']))

        model.train()
        model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext

        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                if cosyvoice_join(group_join, info_dict):
                    break

                if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                else:
                    context = nullcontext

                # 1. Train Discriminator
                with context():
                    batch_dict['turn'] = 'discriminator'
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)
                info_dict = update_parameter_and_lr(model, optimizer_d, scheduler_d, scaler, info_dict)
                optimizer.zero_grad()
                log_per_step(writer, info_dict)

                # 🎯 wandb log for Discriminator
                if 'loss' in info_dict:
                    print(f"[D] Epoch {self.epoch}, Step {self.step}, Batch {batch_idx}, Loss: {info_dict['loss']}")
                    wandb.log({
                        "Discriminator/Loss": info_dict['loss'],
                        "train/step": info_dict["step"],
                        "train/epoch": info_dict["epoch"]
                    })

                # 2. Train Generator
                with context():
                    batch_dict['turn'] = 'generator'
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)
                info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
                optimizer_d.zero_grad()
                log_per_step(writer, info_dict)

                # 🎯 wandb log for Generator + Console log
                if 'loss' in info_dict:
                    loss_dict = info_dict.get('loss_dict', {})
                    loss_total = loss_dict.get('loss', torch.tensor(0.0)).item()
                    loss_gen = loss_dict.get('loss_gen', torch.tensor(0.0)).item()
                    loss_fm = loss_dict.get('loss_fm', torch.tensor(0.0)).item()
                    loss_mel = loss_dict.get('loss_mel', torch.tensor(0.0)).item()
                    loss_tpr = loss_dict.get('loss_tpr', torch.tensor(0.0)).item()
                    loss_f0 = loss_dict.get('loss_f0', torch.tensor(0.0)).item()

                    print(f"[G] Epoch {self.epoch}, Step {self.step}, Batch {batch_idx}, "
                        f"Total: {loss_total:.4f}, Gen: {loss_gen:.4f}, FM: {loss_fm:.4f}, "
                        f"Mel: {loss_mel:.4f}, TPR: {loss_tpr:.4f}, F0: {loss_f0:.4f}")

                    wandb.log({
                        "Generator/Total": loss_total,
                        "Generator/Gen": loss_gen,
                        "Generator/FM": loss_fm,
                        "Generator/Mel": loss_mel,
                        "Generator/TPR": loss_tpr,
                        "Generator/F0": loss_f0,
                        "train/step": info_dict["step"],
                        "train/epoch": info_dict["epoch"]
                    })

                # Checkpoint 저장 조건
                if info_dict['save_per_step'] > 0 and (self.step + 1) % info_dict['save_per_step'] == 0 and \
                (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    dist.barrier()
                    self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                    model.train()

                # step 증가
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1

        dist.barrier()
        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)


    @torch.inference_mode()
    def cv(self, model, cv_data_loader, writer, info_dict, on_batch_end=True):
        ''' Cross validation on
        '''
        logging.info('Epoch {} Step {} on_batch_end {} CV rank {}'.format(self.epoch, self.step + 1, on_batch_end, self.rank))
        model.eval()
        total_num_utts, total_loss_dict = 0, {}  # avoid division by 0
        for batch_idx, batch_dict in enumerate(cv_data_loader):
            info_dict["tag"] = "CV"
            info_dict["step"] = self.step
            info_dict["epoch"] = self.epoch
            info_dict["batch_idx"] = batch_idx

            num_utts = len(batch_dict["utts"])
            total_num_utts += num_utts

            if self.gan is True:
                batch_dict['turn'] = 'generator'
            info_dict = batch_forward(model, batch_dict, None, info_dict)

            for k, v in info_dict['loss_dict'].items():
                if k not in total_loss_dict:
                    total_loss_dict[k] = []
                total_loss_dict[k].append(v.item() * num_utts)
            log_per_step(None, info_dict)
        for k, v in total_loss_dict.items():
            total_loss_dict[k] = sum(v) / total_num_utts
        info_dict['loss_dict'] = total_loss_dict
        log_per_save(writer, info_dict)
        model_name = 'epoch_{}_whole'.format(self.epoch) if on_batch_end else 'epoch_{}_step_{}'.format(self.epoch, self.step + 1)
        save_model(model, model_name, info_dict)
