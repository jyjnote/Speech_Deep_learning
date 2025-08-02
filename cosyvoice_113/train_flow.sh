CUDA_VISIBLE_DEVICES=4 torchrun --nnodes=1 --nproc_per_node=1 \
  --rdzv_id=ljs_flow --rdzv_backend=c10d --rdzv_endpoint=localhost:1234 \
  cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config configs/train_ljspeech.yaml \
  --train_data data/train.data.list \
  --cv_data data/dev.data.list \
  --model flow \
  --model_dir exp/ljspeech/flow \
  --tensorboard_dir tensorboard/ljspeech/flow \
  --ddp.dist_backend nccl \
  --num_workers 2 \
  --prefetch 100 \
  --pin_memory \
  --use_amp
