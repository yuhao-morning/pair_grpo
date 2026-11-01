#!/bin/bash
# 在一台单卡 H100 的 SageMaker 空间上安装：verl-agent（固定提交）+ 我们的补丁 + WebShop 环境。
# 只有 $HOME 在重启后保留，所以 conda 环境和数据都放在 $HOME/pg 下。
# 用法（在服务器上）：bash setup.sh <本仓库 llm 目录的路径>
set -euo pipefail
LLM=${1:?需要本仓库 llm 目录的路径}
ROOT=$HOME/pg
COMMIT=20bd331
mkdir -p $ROOT && cd $ROOT

# 1. verl-agent 固定提交 + 补丁
if [ ! -d verl-agent ]; then git clone -q https://github.com/langfengQ/verl-agent.git; fi
cd verl-agent
git checkout -q $COMMIT
git checkout -q -- .                       # 重装时先还原，补丁总是打在干净的提交上
git apply $LLM/verl_patch/pair.patch
cp $LLM/pair_adv.py gigpo/pair_adv.py

# 2. WebShop 的 conda 环境（verl-agent 要求 Python 3.10 单独一个环境）
eval "$(conda shell.bash hook)"
[ -d $ROOT/env ] || conda create -y -q -p $ROOT/env python=3.10
conda activate $ROOT/env

cd agent_system/environments/env_package/webshop/webshop
if [ ! -d search_engine/indexes_1k ]; then
  ./setup.sh -d small                        # 依赖、openjdk、faiss、1000 商品数据、spaCy 模型、检索索引
fi
cd $ROOT/verl-agent

# 3. 训练栈（版本取自 verl-agent README 的 WebShop 一节）
pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -q flash-attn==2.7.4.post1 --no-build-isolation
pip install -q -e .
pip install -q vllm==0.8.2

python -c "import torch, vllm, flash_attn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'vllm', vllm.__version__)"
python $LLM/test_pair_adv.py
echo "安装完成"
