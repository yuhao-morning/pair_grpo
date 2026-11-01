#!/bin/bash
# 一次 WebShop 训练运行。超参取自 verl-agent examples/gigpo_trainer/run_webshop.sh，
# 只改不影响数学的设置（1 张卡、TP=1、显存比例）。llm/prereg.md 第 1 节。
# 用法：bash run_webshop.sh <臂> <种子> [额外的 hydra 覆盖...]
#   臂：gigpo_bin grpo_bin（Phase 0，二值奖励，复现公布数字）
#       gigpo pair_gigpo pair_grpo grpo pair_step（E1，分级奖励）
set -euo pipefail
ARM=$1; SEED=$2; shift 2
ROOT=$HOME/pg
cd $ROOT/verl-agent
eval "$(conda shell.bash hook)"; conda activate $ROOT/env
ulimit -u 65536 || true
export VLLM_ATTENTION_BACKEND=XFORMERS

case $ARM in
  gigpo_bin)  ADV=gigpo;      GRADED=False ;;
  grpo_bin)   ADV=grpo;       GRADED=False ;;
  gigpo)      ADV=gigpo;      GRADED=True ;;
  pair_gigpo) ADV=pair_gigpo; GRADED=True ;;
  pair_grpo)  ADV=pair_grpo;  GRADED=True ;;
  grpo)       ADV=grpo;       GRADED=True ;;
  pair_step)  ADV=pair_step;  GRADED=True ;;
  *) echo "未知的臂 $ARM"; exit 1 ;;
esac
NAME=${ARM}_s${SEED}
OUT=$ROOT/runs/$NAME
mkdir -p $OUT

python3 -m examples.data_preprocess.prepare --mode text --train_data_size 16 --val_data_size 256 >/dev/null

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM:-0.4} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_norm \
    env.env_name=Webshop \
    env.seed=$SEED \
    env.max_steps=15 \
    env.rollout.n=8 \
    env.resources_per_worker.num_cpus=0.1 \
    env.webshop.graded_reward=$GRADED \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=pair_grpo_webshop \
    trainer.experiment_name=$NAME \
    trainer.default_local_dir=$OUT/ckpt \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ:-150} \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True "$@" 2>&1 | tee -a $OUT/train.log
