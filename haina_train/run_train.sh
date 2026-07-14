#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
CONFIG="${CONFIG:-${SCRIPT_DIR}/config_cpt.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python3.10}"

if [[ -f "${PROJECT_ROOT}/train/MX_env.sh" ]]; then
  source "${PROJECT_ROOT}/train/MX_env.sh"
fi

if [[ "${NPROC_PER_NODE}" == "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/train_haina_cpt.py" --config "${CONFIG}" "$@"
else
  torchrun --nproc_per_node="${NPROC_PER_NODE}" "${SCRIPT_DIR}/train_haina_cpt.py" --config "${CONFIG}" "$@"
fi
