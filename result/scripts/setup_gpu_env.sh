#!/bin/bash
# ============================================================
# MER2026 Track2 重跑 — GPU 机器环境准备脚本
# 在 GPU 机器上执行, 一次性建好 conda 环境 + 装依赖 + 迁移模型
#
# 用法:
#   1) 把 MER2026_Track2 代码 + result 目录拷到 GPU 机器
#      (git clone git@github.com:aa33275111/mer2.git 即可, 权重不入库)
#   2) 把 models/ 目录从本机拷过去 (约 20G):
#      rsync -av /root/MER2026_Track2/models/ user@gpu:/path/MER2026_Track2/models/
#   3) 把数据集目录挂载/拷贝到 GPU 机器, 并修改 config.py 的 DATA_DIR['MER2026']
#   4) 运行: bash result/scripts/setup_gpu_env.sh
# ============================================================
set -e
ENV_NAME="mer2026"
PYTHON="3.10"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
HF_ENDPOINT="https://hf-mirror.com"

echo "========== [1/4] 创建 conda 环境 $ENV_NAME =========="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENV_NAME" python=$PYTHON
conda activate "$ENV_NAME"

echo "========== [2/4] 安装 PyTorch (CUDA 12.1) =========="
# 2026-08-26 验证过的版本组合: 必须用 torch 2.1.2 (torchvision 0.16.2 才有 pytorchvideo 需要的
# functional_tensor; torch 2.3 配 torchvision 0.18 会冲突)。cu121 对应 driver >= 12.1。
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

echo "========== [3/4] 安装项目依赖 (已验证版本) =========="
pip install -i "$PIP_MIRROR" \
  "transformers==4.40.2" "peft==0.11.1" "accelerate==0.30.1" "setuptools<81" "numpy==1.26.4" \
  einops omegaconf decord opencv-python-headless pillow pandas scipy scikit-learn \
  matplotlib librosa soundfile timm ftfy regex zhon pyyaml \
  sentencepiece tiktoken webdataset iopath pytorchvideo \
  openai fire tqdm
# 注意: 千万别让 pip 升级 torch/transformers/peft (会破坏兼容); setuptools<81 是为 pkg_resources

echo "========== [3.5/4] vLLM (OV 标签抽取用, 可选但推荐) =========="
pip install -i "$PIP_MIRROR" "vllm" || echo "[WARN] vllm 安装失败, 可稍后单独装 (reason->OV 标签步骤需要)"

echo "========== [4/4] 校验 =========="
# deepspeed 需要 CUDA_HOME 指向 conda 环境 (含 nvcc)
export CUDA_HOME="$(dirname $(dirname $(which python)))"
python - <<'EOF'
import importlib, warnings; warnings.filterwarnings('ignore')
mods = ['torch','transformers','peft','decord','cv2','einops','omegaconf','webdataset','iopath','timm','pytorchvideo']
for m in mods:
    try:
        importlib.import_module(m); print(f'OK   {m}')
    except Exception as e:
        print(f'MISS {m}: {e}')
import torch, torchvision
print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| CUDA:', torch.cuda.is_available())
import torchvision.transforms.functional_tensor as F_t   # pytorchvideo 兼容性关键
print('functional_tensor OK')
EOF

cat <<'EOT'
========== 完成 ==========
接下来:
1) 改 config.py 的 DATA_DIR['MER2026'] 指向 GPU 机上数据集路径
2) 确认 models/ 已就位 (Qwen2.5-7B-Instruct / chinese-hubert-large / clip-vit-large-patch14 / bert-base-uncased)
3) 先跑小样本冒烟测试 (验证环境+代码):
   CUDA_HOME="$CONDA_PREFIX" CUDA_VISIBLE_DEVICES=0 python result/scripts/sft_smoke_test.py
4) 再跑正式 SFT (2 卡 A100; batch=3/卡, 每轮轻量 val_loss + 每 5 轮生成式 EW-F1):
   torchrun --nproc_per_node=2 --master_port=29500 train.py \
     --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml
   每轮 ckpt 存 output/{cfg}/{job_id}/, 验证指标写 log.txt
5) 训完在 test10 上全量扫查所有 ckpt 挑最优 (见 evaluation.md)
6) GRPO (单卡):
   CUDA_VISIBLE_DEVICES=0 python -u grpo/train_grpo.py \
     --cfg-path=train_configs/grpo_human_ewf1.yaml --sft-ckpt <SFT best ckpt>
EOT
