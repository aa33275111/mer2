#!/bin/bash
# 下载 MER2026 Track2 AffectGPT 所需的 4 个预训练模型到 MER2026_Track2/models/
# 2026-08-25: hf-mirror 对 hf CLI 的请求不稳定 (ConnectTimeout), 改用 huggingface.co 直连
set -u
unset HF_ENDPOINT
export HF_HUB_ENABLE_HF_TRANSFER=0

PY=/root/miniconda3/envs/cosyvoice/bin
MODELS_DIR=/root/MER2026_Track2/models
mkdir -p "$MODELS_DIR"

# 排除无关文件: CoreML / ONNX / TF / Flax / GGUF / 优化格式 (只需 PyTorch 权重)
EXCLUDES=(
  --exclude
  "*.onnx" "*.onnx_data*" "*.mlmodel*" "*.h5" "*.msgpack" "*.gguf" "*.ot" "*.tflite" "*.pb" "*.pbtxt"
)

dl() {
  local repo="$1" dir="$2"
  echo "===== $(date '+%H:%M:%S') 开始 $repo -> $MODELS_DIR/$dir ====="
  for attempt in 1 2 3; do
    "$PY/hf" download "$repo" --local-dir "$MODELS_DIR/$dir" "${EXCLUDES[@]}" 2>&1 | tail -1
    if [ -f "$MODELS_DIR/$dir/config.json" ]; then
      echo "[OK] $dir 完成 (attempt=$attempt)"
      return 0
    fi
    echo "[RETRY] $dir attempt=$attempt 失败, 3s 后重试..."
    sleep 3
  done
  echo "[FAIL] $dir 三次尝试均失败"
  return 1
}

dl "bert-base-uncased" "bert-base-uncased"
dl "openai/clip-vit-large-patch14" "clip-vit-large-patch14"
dl "TencentGameMate/chinese-hubert-large" "chinese-hubert-large"
dl "Qwen/Qwen2.5-7B-Instruct" "Qwen2.5-7B-Instruct"

echo "===== $(date '+%H:%M:%S') 全部结束 ====="
du -sh "$MODELS_DIR"/*
