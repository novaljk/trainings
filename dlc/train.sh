#!/usr/bin/env bash
set -Eeuo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# ---------------------------------------------------------------------------
# Basic paths
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/data}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/out}"
CKPT_DIR="${CKPT_DIR:-${DATA_ROOT}/checkpoints}"

mkdir -p "${OUT_DIR}" "${CKPT_DIR}"

# The training code still uses "../out" and "../checkpoints" from trainer/.
# Link them to the mounted storage so outputs survive container restarts.
link_if_absent() {
    local link_path="$1"
    local target_path="$2"

    if [[ -L "${link_path}" ]]; then
        rm "${link_path}"
        ln -s "${target_path}" "${link_path}"
    elif [[ -e "${link_path}" ]]; then
        echo "[dlc/train.sh] Using existing ${link_path}; not replacing it." >&2
    else
        ln -s "${target_path}" "${link_path}"
    fi
}

link_if_absent "${REPO_ROOT}/out" "${OUT_DIR}"
link_if_absent "${REPO_ROOT}/checkpoints" "${CKPT_DIR}"

# ---------------------------------------------------------------------------
# Training stage
# ---------------------------------------------------------------------------
TRAIN_STAGE="${TRAIN_STAGE:-pretrain}"
case "${TRAIN_STAGE}" in
    pretrain)
        TRAIN_SCRIPT="train_pretrain.py"
        DEFAULT_DATA_PATH="${DATA_ROOT}/pretrain_t2t_mini.jsonl"
        DEFAULT_SAVE_WEIGHT="pretrain"
        DEFAULT_FROM_WEIGHT="none"
        ;;
    sft)
        TRAIN_SCRIPT="train_full_sft.py"
        DEFAULT_DATA_PATH="${DATA_ROOT}/sft_t2t_mini.jsonl"
        DEFAULT_SAVE_WEIGHT="full_sft"
        DEFAULT_FROM_WEIGHT="pretrain"
        ;;
    *)
        echo "[dlc/train.sh] Unsupported TRAIN_STAGE: ${TRAIN_STAGE}. Use pretrain or sft." >&2
        exit 2
        ;;
esac

DATA_PATH="${DATA_PATH:-${DEFAULT_DATA_PATH}}"
SAVE_WEIGHT="${SAVE_WEIGHT:-${DEFAULT_SAVE_WEIGHT}}"
FROM_WEIGHT="${FROM_WEIGHT:-${DEFAULT_FROM_WEIGHT}}"
RESUME="${RESUME:-1}"

if [[ ! -f "${DATA_PATH}" ]]; then
    echo "[dlc/train.sh] Data file not found: ${DATA_PATH}" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Distributed settings
# ---------------------------------------------------------------------------
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-gpu}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

if (( NNODES > 1 )) && [[ "${MASTER_ADDR}" == "127.0.0.1" ]]; then
    echo "[dlc/train.sh] Multi-node training requires MASTER_ADDR to be set." >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
CMD=(
    torchrun
    --nnodes="${NNODES}"
    --nproc_per_node="${NPROC_PER_NODE}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    "${TRAIN_SCRIPT}"
    --data_path "${DATA_PATH}"
    --save_weight "${SAVE_WEIGHT}"
    --from_weight "${FROM_WEIGHT}"
    --from_resume "${RESUME}"
)

if [[ -n "${EPOCHS:-}" ]]; then
    CMD+=(--epochs "${EPOCHS}")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
    CMD+=(--batch_size "${BATCH_SIZE}")
fi
if [[ -n "${LEARNING_RATE:-}" ]]; then
    CMD+=(--learning_rate "${LEARNING_RATE}")
fi
if [[ -n "${HIDDEN_SIZE:-}" ]]; then
    CMD+=(--hidden_size "${HIDDEN_SIZE}")
fi
if [[ -n "${NUM_HIDDEN_LAYERS:-}" ]]; then
    CMD+=(--num_hidden_layers "${NUM_HIDDEN_LAYERS}")
fi
if [[ -n "${MAX_SEQ_LEN:-}" ]]; then
    CMD+=(--max_seq_len "${MAX_SEQ_LEN}")
fi
if [[ -n "${SAVE_INTERVAL:-}" ]]; then
    CMD+=(--save_interval "${SAVE_INTERVAL}")
fi
if [[ -n "${ACCUMULATION_STEPS:-}" ]]; then
    CMD+=(--accumulation_steps "${ACCUMULATION_STEPS}")
fi
if [[ "${USE_WANDB:-0}" == "1" ]]; then
    CMD+=(--use_wandb)
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
    CMD+=(--wandb_project "${WANDB_PROJECT}")
fi

if [[ -n "${TRAIN_ARGS:-}" ]]; then
    read -r -a EXTRA_ARGS <<< "${TRAIN_ARGS}"
    CMD+=("${EXTRA_ARGS[@]}")
fi

cd "${REPO_ROOT}/trainer"
exec "${CMD[@]}"
