# Immutable base: Python, torch 2.4.0, torchvision 0.19.0, CUDA 12.1/cuDNN 9.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime@sha256:68c022c2f4627943a6f3e574cfd2c8ae4256210d5f66ae2b117942e0a8d4fa9d

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    XDG_CACHE_HOME=/tmp/.cache \
    TORCH_EXTENSIONS_DIR=/tmp/torch_extensions

# Arbitrary host UID must be able to create Torch's CUDA kernel cache.
RUN mkdir -p /tmp/.cache/torch/kernels /tmp/torch_extensions \
    && chmod -R a+rwx /tmp/.cache /tmp/torch_extensions

# The base digest locks inherited packages. This lock adds/replaces precisely
# the distributions selected on that base; no dependency resolution at build time.
COPY models/docker/requirements.lock /app/requirements.lock
RUN python -m pip install --no-cache-dir --no-deps --require-hashes -r /app/requirements.lock \
    && python -m pip check \
    && python -m pip freeze --all > /app/environment.freeze.txt

COPY models/ /app/models/
COPY README.md /app/README.md
RUN python -m unittest discover -s models/tests -v \
    && python -c "from scipy.optimize import linear_sum_assignment; from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Owlv2ForObjectDetection, Qwen3VLForConditionalGeneration; print('All three model classes and scipy import successfully')" \
    && python models/docker/build_info.py

LABEL org.opencontainers.image.title="butterfly-detection-models" \
      org.opencontainers.image.description="Local butterfly/moth detection: Grounding DINO tiny, OWLv2 base, Qwen3-VL-2B-Instruct"

CMD ["python", "-m", "models", "--help"]
