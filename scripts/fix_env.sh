#!/usr/bin/env bash
set -euo pipefail

echo "[fix_env] Activating conda env: automl"
if [[ -f /root/autodl-tmp/miniconda3/bin/activate ]]; then
  # Preferred path in this workspace
  source /root/autodl-tmp/miniconda3/bin/activate automl
elif [[ -f /root/miniconda3/bin/activate ]]; then
  # Fallback path
  source /root/miniconda3/bin/activate automl
else
  echo "[fix_env] ERROR: conda activate script not found."
  exit 1
fi

python -m pip install --upgrade pip

echo "[fix_env] Checking core dependencies..."
core_missing=$(python - <<'PY'
import importlib.util
required = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "joblib": "joblib",
    "lightgbm": "lightgbm",
    "optuna": "optuna",
    "transformers": "transformers",
    "datasets": "datasets",
    "tokenizers": "tokenizers",
    "huggingface-hub": "huggingface_hub",
    "accelerate": "accelerate",
    "sentencepiece": "sentencepiece",
    "timm": "timm",
    "nltk": "nltk",
    "nlpaug": "nlpaug",
    "colorlog": "colorlog",
    "librosa": "librosa",
    "soundfile": "soundfile",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
}
missing = [pkg for pkg, mod in required.items() if importlib.util.find_spec(mod) is None]
print(" ".join(missing))
PY
)
if [[ -n "${core_missing}" ]]; then
  echo "[fix_env] Installing core missing packages: ${core_missing}"
  python -m pip install ${core_missing}
else
  echo "[fix_env] Core dependencies already installed."
fi

echo "[fix_env] Checking optional dependencies..."
optional_missing=$(python - <<'PY'
import importlib.util
optional = {
    "torchcodec": "torchcodec",
    "torch-audiomentations": "torch_audiomentations",
}
missing = [pkg for pkg, mod in optional.items() if importlib.util.find_spec(mod) is None]
print(" ".join(missing))
PY
)
if [[ -n "${optional_missing}" ]]; then
  echo "[fix_env] Installing optional packages: ${optional_missing}"
  for pkg in ${optional_missing}; do
    if ! python -m pip install "${pkg}"; then
      echo "[fix_env] WARN: failed to install ${pkg}. You may need to install it manually."
    fi
  done
else
  echo "[fix_env] Optional dependencies already installed."
fi

echo "[fix_env] Ensuring PyTorch is available..."
python - <<'PY'
try:
    import torch
    print(f"[fix_env] torch={torch.__version__}")
except Exception as exc:
    raise SystemExit(f"[fix_env] ERROR: PyTorch not available: {exc}")
PY

echo "[fix_env] Installing system libs (ffmpeg/libsndfile) via conda if available..."
if command -v conda >/dev/null 2>&1; then
  conda install -y -c conda-forge ffmpeg libsndfile || true
else
  echo "[fix_env] WARN: conda not found in PATH; skip ffmpeg/libsndfile."
fi

echo "[fix_env] Downloading NLTK resources..."
python - <<'PY'
import nltk
resources = [
    "averaged_perceptron_tagger_eng",
    "punkt",
    "wordnet",
    "omw-1.4",
]
for res in resources:
    nltk.download(res, quiet=False)
PY

if [[ "${INSTALL_TF:-0}" == "1" ]]; then
  echo "[fix_env] Optional: installing TensorFlow (INSTALL_TF=1)"
  python -m pip install "tensorflow>=2.12"
fi

echo "[fix_env] Allow HuggingFace online access in current shell..."
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

echo "[fix_env] Done."
