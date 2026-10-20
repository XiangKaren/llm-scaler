#!/usr/bin/env bash
# Launch an OFFLINE fp8 checkpoint (pre-quantized, 128x128 block scales) on Intel BMG, TP=2:
#   Qwen3.6-35B-A3B-FP8 (MoE) and Qwen3.8-27B-FP8 (dense), both hybrid GDN + full attention.
# For the bf16 checkpoint quantized online to fp8, use start_qwen3_6_service.sh instead.
#
# Run INSIDE the container (sglang 0.5.13-dev + sgl-kernel-xpu dev-bmg):
#   MODEL_PATH=/models/Qwen3.6-35B-A3B-FP8 bash scripts/start_qwen3_6_fp8_offline_service.sh
#   MODEL_PATH=/models/Qwen3.8-27B-FP8     bash scripts/start_qwen3_6_fp8_offline_service.sh
#
# Memory: RESERVE_GB (default 12) is left free on each card for other services. The static
# pool (weights + KV + mamba) gets the rest, i.e. --mem-fraction-static = 1 - RESERVE_GB/VRAM.
# The SSM state stays fp32 (the checkpoints' mamba_ssm_dtype); that is twice the mamba pool of
# the online recipe, which forces fp16. So the mamba slot count is set low enough that the KV
# pool still holds one full CONTEXT_LENGTH request. The slot count also caps concurrency:
#   MAMBA_STRATEGY=extra_buffer (default; radix prefix cache + HiCache possible): 4 slots/request
#     35B-A3B-FP8: 32 slots -> 8 concurrent, ~175k KV tokens at 12 GB reserved
#     27B-FP8    : 16 slots -> 4 concurrent, ~139k KV tokens at 12 GB reserved
#   MAMBA_STRATEGY=no_buffer (radix cache off, no HiCache): 1 slot/request
#     35B-A3B-FP8: 32 slots -> 32 concurrent;  27B-FP8: 10 slots -> 10 concurrent
#
# Optional: MAMBA_STRATEGY=extra_buffer|no_buffer TP_SIZE=2 PORT=30000 HOST=0.0.0.0 RESERVE_GB=12 MEM_FRACTION_STATIC=<override>
#           MAX_MAMBA_CACHE_SIZE=<override> CONTEXT_LENGTH=132096 CHUNKED_PREFILL_SIZE=<override>
# HiCache (L2 host tier): HICACHE=1 [HICACHE_RATIO=3 HICACHE_WRITE_POLICY=write_through]
#   The host pool is HICACHE_RATIO x the device pool, so 3 keeps three 128K prefixes in host RAM.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-35B-A3B-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-2}"
RESERVE_GB="${RESERVE_GB:-12}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-132096}"

# Loopback must bypass any proxy: the server polls itself over HTTP during startup.
export no_proxy="127.0.0.1,localhost,0.0.0.0${no_proxy:+,${no_proxy}}"
export NO_PROXY="${no_proxy}"

# --- device selection ---
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0,1}"

# --- per-model defaults (validated with RESERVE_GB=12, CONTEXT_LENGTH=132096) ---
MAMBA_STRATEGY="${MAMBA_STRATEGY:-extra_buffer}"
case "${MODEL_PATH,,}:${MAMBA_STRATEGY}" in
    *27b*:no_buffer) DEF_MAMBA=10; DEF_CHUNK=2048 ;;
    *27b*:*)         DEF_MAMBA=16; DEF_CHUNK=2048 ;;
    *)               DEF_MAMBA=32; DEF_CHUNK=4096 ;;
esac
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-${DEF_MAMBA}}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-${DEF_CHUNK}}"

if [[ -z "${MEM_FRACTION_STATIC:-}" ]]; then
    MEM_FRACTION_STATIC=$(python3 - "${RESERVE_GB}" <<'PY'
import sys, torch
total_gb = torch.xpu.get_device_properties(0).total_memory / 2**30
print(f"{1 - float(sys.argv[1]) / total_gb:.2f}")
PY
)
fi
echo "VRAM reserve ${RESERVE_GB} GB -> --mem-fraction-static ${MEM_FRACTION_STATIC}," \
     "--max-mamba-cache-size ${MAX_MAMBA_CACHE_SIZE}, --context-length ${CONTEXT_LENGTH}" >&2

# --- SSM state: fp32, as the checkpoint asks. Never inherit the online recipe's fp16 override. ---
unset SGLANG_MAMBA_SSM_DTYPE SGLANG_MAMBA_CONV_DTYPE

# --- offline block-fp8 kernels ---
# Routes the fp8 MoE experts to sgl-kernel's Xe2 W8A16 grouped GEMM (block scales). It takes
# bf16 activations only, hence --dtype bfloat16. Dense fp8 linears use the oneDNN W8A16 GEMM.
export SGLANG_USE_SGL_XPU=1
# GDN decode conv (fused_seq) and GDN chunked prefill on ESIMD: compute-only kernels that never
# read the fp8 weights. The extend kernel is the prefill TTFT lever.
export SGL_XPU_GDN_ESIMD=1
export SGL_XPU_GDN_EXTEND_ESIMD=1
# Full-attention decode on the SYCL FMHA kernel: flat decode speed up to 128K.
export SGL_XPU_DECODE_SGLANG_ATTN=0
# XPU graph off (accuracy), decode runs eager.
export SGL_XPU_ENABLE_GRAPH=0
# Do NOT enable the online recipe's ESIMD fusions (SGL_XPU_ESIMD_MOE*, *_RESADD_NORM,
# GDN_NORM_GEMV, MOE_ROUTER_FP8, ...): they read per-tensor e5m2 weights, not block scales.

OPTIONAL_ARGS=()
if [[ "${MAMBA_STRATEGY}" == no_buffer ]]; then
    # no_buffer cannot track mamba state at page_size 64, so the radix prefix cache goes off.
    [[ "${HICACHE:-0}" == 1 ]] && { echo "HICACHE=1 needs MAMBA_STRATEGY=extra_buffer" >&2; exit 2; }
    OPTIONAL_ARGS+=(--disable-radix-cache)
fi
if [[ "${HICACHE:-0}" == 1 ]]; then
    OPTIONAL_ARGS+=(
        --enable-hierarchical-cache
        --hicache-ratio "${HICACHE_RATIO:-3}"
        --hicache-write-policy "${HICACHE_WRITE_POLICY:-write_through}"
        --hicache-io-backend kernel
    )
fi

# --mamba-scheduler-strategy extra_buffer + --page-size 64 keep the radix prefix cache on.
# --skip-server-warmup: the stock warmup sends an image, which hits an unrelated dynamo error
# in the vision tower.
exec python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --tp "${TP_SIZE}" \
    --dtype bfloat16 \
    --quantization fp8 \
    --attention-backend intel_xpu \
    --trust-remote-code \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --mamba-ssm-dtype float32 \
    --max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}" \
    --page-size 64 \
    --mamba-scheduler-strategy "${MAMBA_STRATEGY}" \
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
    --context-length "${CONTEXT_LENGTH}" \
    --disable-overlap-schedule \
    --disable-cuda-graph \
    --skip-server-warmup \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-cache-report \
    --enable-metrics \
    --host "${HOST}" \
    --port "${PORT}" \
    ${OPTIONAL_ARGS[@]+"${OPTIONAL_ARGS[@]}"}
