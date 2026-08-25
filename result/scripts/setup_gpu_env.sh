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
# 根据 GPU 机器实际 CUDA 版本选择; cu121 对应 NVIDIA driver >= 525
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

echo "========== [3/4] 安装项目依赖 =========="
pip install -i "$PIP_MIRROR" \
  transformers==4.51.3 peft accelerate einops omegaconf \
  decord opencv-python-headless pillow numpy pandas scipy scikit-learn \
  matplotlib librosa soundfile torchaudio timm ftfy regex zhon pyyaml \
  sentencepiece tiktoken webdataset iopath pytorchvideo \
  openai fire tqdm

echo "========== [3.5/4] vLLM (OV 标签抽取用, 可选但推荐) =========="
pip install -i "$PIP_MIRROR" "vllm" || echo "[WARN] vllm 安装失败, 可稍后单独装 (reason->OV 标签步骤需要)"

echo "========== [4/4] 校验 =========="
python - <<'EOF'
import importlib, warnings; warnings.filterwarnings('ignore')
mods = ['torch','transformers','peft','decord','cv2','einops','omegaconf','webdataset','iopath','timm','vllm']
for m in mods:
    try:
        importlib.import_module(m); print(f'OK   {m}')
    except Exception as e:
        print(f'MISS {m}: {e}')
import torch
print('CUDA available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
EOF

cat <<'EOT'
========== 完成 ==========
接下来:
1) 改 config.py 的 DATA_DIR['MER2026'] 指向 GPU 机上数据集路径
2) 确认 models/ 已就位 (Qwen2.5-7B-Instruct / chinese-hubert-large / clip-vit-large-patch14 / bert-base-uncased)
3) 先跑通 SFT (4 卡 A100, 与历史配置 4GPU×3batch 一致):
   torchrun --nproc_per_node=4 --master_port=29500 train.py \
     --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml
4) 然后 GRPO (单卡):
   CUDA_VISIBLE_DEVICES=0 python -u grpo/train_grpo.py \
     --cfg-path=train_configs/grpo_human_ewf1.yaml
5) 评测流程见 my_affectgpt/evaluation/evaluation.md
EOT
