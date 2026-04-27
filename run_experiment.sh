#!/usr/bin/env bash
# Reproduce a single experiment end-to-end.
#
# Pick the experiment with TARGET:
#   gbm  — GBM forward process (alpha=1, log-prices). Paper's headline.
#   vp   — VP SDE baseline on log-returns (alpha=0).
#   ve   — VE SDE baseline on log-returns (alpha=0).
#
# Default: TARGET=gbm + exponential schedule — the paper's best heavy-tail fit
# (§4: GBM-exponential α≈4.62 vs. empirical α≈4.35). Each TARGET sets sensible
# defaults for ALPHA / SDE / SCHEDULE; override any of those env vars to deviate.
#
# Examples:
#   ./run_experiment.sh                          # paper headline (GBM-exponential)
#   ./run_experiment.sh -e 2                     # quick smoke (2 epochs)
#   ./run_experiment.sh --epochs 200             # 200 epochs (long form)
#   ./run_experiment.sh -b 16                    # smaller batch (e.g. tight VRAM)
#   ./run_experiment.sh --batch_size 8 -e 2      # combine flags
#   TARGET=ve ./run_experiment.sh                # VE baseline, exponential schedule
#   TARGET=vp ./run_experiment.sh                # VP baseline, cosine schedule
#   TARGET=gbm SCHEDULE=cosine ./run_experiment.sh   # GBM with cosine schedule instead
#   EPOCHS=2 BATCH_SIZE=16 STEPS=100 NUM_SAMPLES=5 ./run_experiment.sh   # quick smoke (env)
#
# Idempotency:
#   - data prep is skipped if data/sp500_subseq*.pt already exist
#   - training auto-resumes from the highest-numbered checkpoint
#   - sampling and plotting overwrite their output folders
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_experiment.sh [-e|--epochs N] [-b|--batch_size N] [-g|--gpus N] [-h|--help]

Flags:
  -e, --epochs N        Number of training epochs (default: 1000, env: EPOCHS).
  -b, --batch_size N    Per-rank training batch size (default: 64, env: BATCH_SIZE).
                        Effective batch under DDP is batch_size * gpus.
                        Lower (16, 8) if the run hits CUDNN_STATUS_NOT_INITIALIZED
                        (usually GPU OOM at workspace allocation, not a real bug).
  -g, --gpus N          Number of GPUs for training (default: 1, env: NUM_GPUS).
                        N=1 -> plain `python`. N>1 -> `torchrun --nproc_per_node=N`
                        (DDP). Sampling is always single-GPU.

All other knobs are env-var only:
  TARGET, ALPHA, SDE, SCHEDULE, LR, SIGMA_MIN, SIGMA_MAX,
  NUM_SAMPLES, STEPS, SNR, N_CORR, CUDA_VISIBLE_DEVICES.

Flag values take precedence over env vars; env vars take precedence over defaults.
EOF
}

# --- CLI flag parsing ----------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--epochs)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 1; }
      EPOCHS="$2"; shift 2 ;;
    -b|--batch_size|--batch-size)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 1; }
      BATCH_SIZE="$2"; shift 2 ;;
    -g|--gpus)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 1; }
      NUM_GPUS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1 (try -h for help)" >&2; exit 1 ;;
  esac
done

# --- Forgive a common typo: SCHEDULER -> SCHEDULE -----------------------
if [[ -n "${SCHEDULER:-}" ]]; then
  if [[ -z "${SCHEDULE:-}" ]]; then
    echo "Warning: env var SCHEDULER detected — treating as SCHEDULE='${SCHEDULER}' (the canonical name is SCHEDULE)." >&2
    SCHEDULE="${SCHEDULER}"
  else
    echo "Warning: both SCHEDULE='${SCHEDULE}' and SCHEDULER='${SCHEDULER}' set; ignoring SCHEDULER." >&2
  fi
fi

# --- Target selection ----------------------------------------------------
TARGET="${TARGET:-gbm}"   # gbm | vp | ve

case "${TARGET}" in
  gbm)
    ALPHA="${ALPHA:-1}"
    # Per paper §3 (eq. 3.5): with drift μ_t = ½σ_t² the GBM forward
    # process reduces to a VE SDE in log-price space — hence VE here.
    SDE="${SDE:-VE}"
    # Paper §4: GBM-exponential is the closest fit to empirical heavy tails
    # (α≈4.62 vs. empirical α≈4.35). cosine and linear are also valid choices.
    SCHEDULE="${SCHEDULE:-exponential}"
    ;;
  vp)
    ALPHA="${ALPHA:-0}"
    SDE="${SDE:-VP}"
    SCHEDULE="${SCHEDULE:-cosine}"
    ;;
  ve)
    ALPHA="${ALPHA:-0}"
    SDE="${SDE:-VE}"
    SCHEDULE="${SCHEDULE:-exponential}"
    ;;
  *)
    echo "Unknown TARGET=${TARGET}; valid: gbm, vp, ve" >&2
    exit 1
    ;;
esac

# --- Other knobs ---------------------------------------------------------
SIGMA_MIN="${SIGMA_MIN:-0.01}"
SIGMA_MAX="${SIGMA_MAX:-1.0}"

# Training (paper §4 defaults)
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-1000}"
LR="${LR:-1e-4}"
NUM_GPUS="${NUM_GPUS:-1}"

# Sampling (paper §4 defaults)
NUM_SAMPLES="${NUM_SAMPLES:-120}"
STEPS="${STEPS:-2000}"
SNR="${SNR:-0.2}"
N_CORR="${N_CORR:-1}"

# --- Auto-derived paths (mirrors train.py's preset logic) ----------------
# Schedule is baked into MODEL_DIR so different schedules don't collide.
if [[ "${ALPHA}" == "1" ]]; then
  MODEL_DIR="save_model_bs_${SCHEDULE}_64"
  TAG="BS_${SCHEDULE}"
elif [[ "${ALPHA}" == "0" && "${SDE}" == "VE" ]]; then
  MODEL_DIR="save_model_ve_${SCHEDULE}_64"
  TAG="VE_${SCHEDULE}"
elif [[ "${ALPHA}" == "0" && "${SDE}" == "VP" ]]; then
  MODEL_DIR="save_model_vp_${SCHEDULE}_64"
  TAG="VP_${SCHEDULE}"
else
  echo "Unsupported (ALPHA, SDE) = (${ALPHA}, ${SDE})" >&2
  exit 1
fi

PLOT_DIR="${TAG}_plot_"
CSV_DIR="${TAG}_csv_"
TS_DIR="${TAG}_timeseries"

# --- Setup ---------------------------------------------------------------
banner() { printf '\n=== %s ===\n' "$1"; }

banner "Activating venv"
# shellcheck disable=SC1091
source .venv/bin/activate

# --- 1. Data prep --------------------------------------------------------
banner "Data prep"
if [[ -f data/sp500_subseq.pt && -f data/sp500_subseq_log.pt && -f data/sp500_subseq_original.pt ]]; then
  echo "data/sp500_subseq*.pt already present — skipping download."
else
  mkdir -p data
  python data_download.py
  mv sp500_subseq*.pt global_*.pkl data/
fi

# --- 2. Training ---------------------------------------------------------
# Multi-GPU via torchrun when NUM_GPUS > 1; plain python otherwise.
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  TRAIN_LAUNCHER=(torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}")
  EFFECTIVE_BATCH=$((BATCH_SIZE * NUM_GPUS))
  banner "Training: target=${TARGET} (alpha=${ALPHA}, sde=${SDE}, schedule=${SCHEDULE}) epochs=${EPOCHS} | DDP nproc=${NUM_GPUS} effective_batch=${EFFECTIVE_BATCH}"
  # NCCL P2P over PCIe hangs on this multi-tenant host — the first all_reduce
  # never returns. Disable P2P so NCCL uses SHM/sockets instead.
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
else
  TRAIN_LAUNCHER=(python)
  banner "Training: target=${TARGET} (alpha=${ALPHA}, sde=${SDE}, schedule=${SCHEDULE}) epochs=${EPOCHS} | single-GPU batch=${BATCH_SIZE}"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${TRAIN_LAUNCHER[@]}" train.py \
    --alpha "${ALPHA}" --sde "${SDE}" --noise_schedule "${SCHEDULE}" \
    --batch_size "${BATCH_SIZE}" --epochs "${EPOCHS}" --lr "${LR}" \
    --sigma_min "${SIGMA_MIN}" --sigma_max "${SIGMA_MAX}"

# Pick the highest-numbered checkpoint that actually exists.
CKPT="$(ls -1 "${MODEL_DIR}"/model_epoch_*.pth 2>/dev/null \
        | sed 's/.*model_epoch_\([0-9]*\)\.pth/\1 &/' \
        | sort -n -k1,1 \
        | tail -1 \
        | awk '{print $2}')"
if [[ -z "${CKPT}" ]]; then
  echo "No checkpoint produced under ${MODEL_DIR}/" >&2
  exit 1
fi
echo "Using checkpoint: ${CKPT}"

# --- 3. Sampling + stylized-fact plots -----------------------------------
banner "Sampling ${NUM_SAMPLES} sequences (${STEPS} reverse-diffusion steps)"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python plot_result.py \
    --model_path "${CKPT}" \
    --alpha "${ALPHA}" --sde "${SDE}" --noise_schedule "${SCHEDULE}" \
    --sigma_min "${SIGMA_MIN}" --sigma_max "${SIGMA_MAX}" \
    --num_samples "${NUM_SAMPLES}" --steps "${STEPS}" --snr "${SNR}" --n_corr "${N_CORR}" \
    --plot_folder "${PLOT_DIR}" \
    --csv_folder "${CSV_DIR}" \
    --plot_timeseries "${TS_DIR}"

# --- 4. Real-data baseline (only if not already there) -------------------
banner "Real-data baseline"
if [[ -d ori_plot && -n "$(ls -A ori_plot 2>/dev/null)" ]]; then
  echo "ori_plot/ already populated — skipping."
else
  python ori_plot_result.py \
      --data_file data/sp500_subseq_original.pt \
      --num_samples 1000
fi

# --- 5. Done -------------------------------------------------------------
banner "Done"
cat <<EOF
Synthetic stylized-fact plots:  ${PLOT_DIR}/
Per-sample log-return CSVs:     ${CSV_DIR}/
Recovered price series PNGs:    ${TS_DIR}/
Real-data baseline plots:       ori_plot/

Compare the matching pair of files in ${PLOT_DIR}/ and ori_plot/ side by side:
  generated_metrics_heavy_tail.png
  generated_metrics_volatility_clustering.png
  generated_metrics_leverage_effect.png
  generated_metrics_linear_unpredictability.png
  generated_metrics_coarse_fine.png
  generated_metrics_gain_loss_asymmetry.png
EOF
