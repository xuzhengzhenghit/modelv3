#!/usr/bin/env bash
set -euo pipefail

# ── HTML online render CPT training launcher ──
# Usage:
#   NPROC_PER_NODE=1 bash run_cpt_html.sh              # single GPU
#   NPROC_PER_NODE=8 bash run_cpt_html.sh              # 8 GPU DDP
#   CONFIG=config_cpt_html_stage2.yaml bash run_cpt_html.sh  # custom config

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/data/haina_html_render:${PYTHONPATH:-}"
CONFIG="${CONFIG:-${SCRIPT_DIR}/config_cpt_html_stage1.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python3.10}"

# MACA environment
if [[ -f "${SCRIPT_DIR}/MX_env.sh" ]]; then
  source "${SCRIPT_DIR}/MX_env.sh"
fi

echo "============================================"
echo " CPT HTML Render Training"
echo " config   : ${CONFIG}"
echo " gpus     : ${NPROC_PER_NODE}"
echo "============================================"

if [[ "${NPROC_PER_NODE}" == "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/train_haina_cpt.py" --config "${CONFIG}" "$@"
else
  torchrun --nproc_per_node="${NPROC_PER_NODE}" "${SCRIPT_DIR}/train_haina_cpt.py" --config "${CONFIG}" "$@"
fi
