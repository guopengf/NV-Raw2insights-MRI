#!/bin/bash
set -u -o pipefail

REPO_ROOT=/workspace/code/NV-Raw2insights-MRI-fork
BASE_CONFIG=configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_pg.json
BENCH_TOOL=scripts/tools/benchmark_4dflow_dataloader.py
RUN_ID=${1:?Usage: run_4dflow_dataloader_sweep.sh RUN_ID}
RUN_ROOT_REL=outputs/4dflow/dataloader_tuning_small6_flowvn/${RUN_ID}
RUN_ROOT=${REPO_ROOT}/${RUN_ROOT_REL}
NODE_RANK=${SLURM_PROCID:-0}
WORLD_NODES=${SLURM_NTASKS:-2}
MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}
TRIAL_INDEX=0

export MASTER_ADDR
export HF_HOME=${REPO_ROOT}/cache
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export WANDB_SILENT=true
export PYTHONUNBUFFERED=1
export NCCL_P2P_LEVEL=NVL
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DESYNC_DEBUG=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONDA_SH=~/miniconda3/etc/profile.d/conda.sh
if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Missing Conda shell hook: ${CONDA_SH}" >&2
    exit 1
fi
source "${CONDA_SH}" || exit 1
conda activate nv-raw2insights-mri || exit 1
cd "${REPO_ROOT}"

wait_for_file() {
    local path=$1
    local timeout_seconds=${2:-300}
    local elapsed=0
    while [[ ! -f "${path}" ]]; do
        sleep 1
        elapsed=$((elapsed + 1))
        if (( elapsed >= timeout_seconds )); then
            echo "Timed out waiting for ${path}" >&2
            return 1
        fi
    done
}

if (( NODE_RANK == 0 )); then
    mkdir -p "${RUN_ROOT}/control" "${RUN_ROOT}/logs"
    {
        echo "run_id=${RUN_ID}"
        echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
        echo "master_addr=${MASTER_ADDR}"
        echo "world_nodes=${WORLD_NODES}"
        echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "base_config_sha256=$(sha256sum "${BASE_CONFIG}" | awk '{print $1}')"
        echo "train_py_sha256=$(sha256sum scripts/train.py | awk '{print $1}')"
        echo "train_utils_py_sha256=$(sha256sum scripts/train_utils.py | awk '{print $1}')"
        echo "utils_py_sha256=$(sha256sum scripts/utils.py | awk '{print $1}')"
        echo "container_image=/home/pengfeig/workspace/cache/conda_raw2insights.sqsh"
    } > "${RUN_ROOT}/provenance.env"
    scontrol show job "${SLURM_JOB_ID}" > "${RUN_ROOT}/slurm_job.txt"
    touch "${RUN_ROOT}/control/run_ready"
fi
wait_for_file "${RUN_ROOT}/control/run_ready"

start_telemetry() {
    local trial_dir=$1
    local host
    host=$(hostname)
    mkdir -p "${trial_dir}/telemetry" "${trial_dir}/logs"
    local gpu_file="${trial_dir}/telemetry/gpu_${host}.csv"
    local host_file="${trial_dir}/telemetry/host_${host}.csv"
    echo "timestamp,index,utilization_gpu_percent,memory_used_mib,power_draw_w" > "${gpu_file}"
    echo "timestamp,mem_available_bytes,shm_available_bytes,load1" > "${host_file}"
    (
        while true; do
            local_timestamp=$(date +%s.%N)
            nvidia-smi \
                --query-gpu=index,utilization.gpu,memory.used,power.draw \
                --format=csv,noheader,nounits 2>/dev/null \
                | awk -v timestamp="${local_timestamp}" -F',' \
                    '{gsub(/^ +| +$/, "", $1); gsub(/^ +| +$/, "", $2); gsub(/^ +| +$/, "", $3); gsub(/^ +| +$/, "", $4); print timestamp "," $1 "," $2 "," $3 "," $4}' \
                >> "${gpu_file}"
            sleep 2
        done
    ) &
    GPU_TELEMETRY_PID=$!
    (
        while true; do
            local_timestamp=$(date +%s.%N)
            mem_available=$(awk '/MemAvailable:/ {print $2 * 1024}' /proc/meminfo)
            shm_available=$(df -B1 --output=avail /dev/shm | tail -n 1 | tr -d ' ')
            load1=$(awk '{print $1}' /proc/loadavg)
            echo "${local_timestamp},${mem_available},${shm_available},${load1}" >> "${host_file}"
            sleep 2
        done
    ) &
    HOST_TELEMETRY_PID=$!
}

stop_telemetry() {
    kill "${GPU_TELEMETRY_PID}" "${HOST_TELEMETRY_PID}" 2>/dev/null || true
    wait "${GPU_TELEMETRY_PID}" "${HOST_TELEMETRY_PID}" 2>/dev/null || true
}

run_trial() {
    local phase=$1
    local tag=$2
    local workers=$3
    local prefetch=$4
    local pin_memory=$5
    local omp_threads=$6
    local max_batches=$7
    local warmup_batches=$8
    local seed=$9
    local detailed=${10}
    local trial_dir="${RUN_ROOT}/trials/${tag}"
    local ready_file="${RUN_ROOT}/control/${tag}.config_ready"
    local summary_file="${RUN_ROOT}/control/${tag}.summary_ready"
    local config_rel="${RUN_ROOT_REL}/trials/${tag}/effective_config.json"
    local detailed_args=()
    local timeout_minutes=25

    TRIAL_INDEX=$((TRIAL_INDEX + 1))
    if (( max_batches >= 170 )); then
        timeout_minutes=45
    fi
    if (( detailed == 1 )); then
        detailed_args+=(--detailed)
    fi

    if (( NODE_RANK == 0 )); then
        python "${BENCH_TOOL}" make-config \
            --base-config "${BASE_CONFIG}" \
            --repo-root "${REPO_ROOT}" \
            --run-root "${RUN_ROOT_REL}" \
            --tag "${tag}" \
            --phase "${phase}" \
            --workers "${workers}" \
            --prefetch "${prefetch}" \
            --pin-memory "${pin_memory}" \
            --omp-threads "${omp_threads}" \
            --max-batches "${max_batches}" \
            --warmup-batches "${warmup_batches}" \
            --seed "${seed}" \
            "${detailed_args[@]}"
        touch "${ready_file}"
    fi
    wait_for_file "${ready_file}"

    echo "[trial_start] node=${NODE_RANK} phase=${phase} tag=${tag} workers=${workers} prefetch=${prefetch} pin=${pin_memory} omp=${omp_threads} batches=${max_batches}"
    start_telemetry "${trial_dir}"
    OMP_NUM_THREADS="${omp_threads}" MKL_NUM_THREADS="${omp_threads}" \
        timeout --signal=TERM --kill-after=60s "${timeout_minutes}m" \
        torchrun \
            --nproc_per_node=8 \
            --nnodes="${WORLD_NODES}" \
            --master_addr="${MASTER_ADDR}" \
            --master_port="$((27000 + TRIAL_INDEX))" \
            --node_rank="${NODE_RANK}" \
            scripts/train.py \
            --config "${config_rel}" \
            > "${trial_dir}/logs/node_${NODE_RANK}.log" 2>&1
    local exit_code=$?
    stop_telemetry
    echo "${exit_code}" > "${trial_dir}/exit_code_node${NODE_RANK}.txt"

    if (( NODE_RANK == 0 )); then
        wait_for_file "${trial_dir}/exit_code_node1.txt" 120
        python "${BENCH_TOOL}" summarize --trial-dir "${trial_dir}" \
            > "${trial_dir}/summary_stdout.log" 2>&1 || true
        touch "${summary_file}"
    fi
    wait_for_file "${summary_file}" 180
    echo "[trial_end] node=${NODE_RANK} tag=${tag} exit_code=${exit_code}"
    return 0
}

rank_trials() {
    local phases=$1
    local top=$2
    local min_runs=$3
    local unique_workers=$4
    local output=$5
    local unique_args=()
    local ready_file="${output}.ready"
    if (( unique_workers == 1 )); then
        unique_args+=(--unique-workers)
    fi
    if (( NODE_RANK == 0 )); then
        python "${BENCH_TOOL}" rank \
            --run-root "${RUN_ROOT}" \
            --phases "${phases}" \
            --top "${top}" \
            --min-runs "${min_runs}" \
            --output "${output}" \
            "${unique_args[@]}"
        touch "${ready_file}"
    fi
    wait_for_file "${ready_file}"
}

# Detailed attribution run. It is not included in throughput rankings.
run_trial diagnostic diagnostic_w4_pf2_pin1_omp1 4 2 1 1 48 8 20260813 1

# Coarse worker-count sweep with the current prefetch and pinning behavior.
run_trial coarse coarse_w0_pf1_pin1_omp1 0 1 1 1 72 12 20260813 0
run_trial coarse coarse_w1_pf2_pin1_omp1 1 2 1 1 72 12 20260813 0
run_trial coarse coarse_w2_pf2_pin1_omp1 2 2 1 1 72 12 20260813 0
run_trial coarse coarse_w4_pf2_pin1_omp1 4 2 1 1 72 12 20260813 0
run_trial coarse coarse_w8_pf2_pin1_omp1 8 2 1 1 72 12 20260813 0
run_trial coarse coarse_w12_pf2_pin1_omp1 12 2 1 1 72 12 20260813 0

# Refine prefetch for the two best positive worker counts.
TOP_WORKERS="${RUN_ROOT}/control/top_workers.tsv"
rank_trials coarse 2 1 1 "${TOP_WORKERS}"
while read -r workers _prefetch _pin _omp _score _source; do
    run_trial refine "refine_w${workers}_pf1_pin1_omp1" "${workers}" 1 1 1 72 12 20260813 0
    run_trial refine "refine_w${workers}_pf4_pin1_omp1" "${workers}" 4 1 1 72 12 20260813 0
done < "${TOP_WORKERS}"

# Test pinned versus pageable host memory for the best two settings.
TOP_PAIRS="${RUN_ROOT}/control/top_pairs.tsv"
rank_trials coarse,refine 2 1 0 "${TOP_PAIRS}"
while read -r workers prefetch _pin omp_threads _score _source; do
    run_trial pin "pin_w${workers}_pf${prefetch}_pin0_omp${omp_threads}" \
        "${workers}" "${prefetch}" 0 "${omp_threads}" 72 12 20260813 0
done < "${TOP_PAIRS}"

# Test two CPU threads per worker only for the best current setting.
TOP_FOR_OMP="${RUN_ROOT}/control/top_for_omp.tsv"
rank_trials coarse,refine,pin 1 1 0 "${TOP_FOR_OMP}"
while read -r workers prefetch pin_memory _omp _score _source; do
    run_trial omp "omp_w${workers}_pf${prefetch}_pin${pin_memory}_omp2" \
        "${workers}" "${prefetch}" "${pin_memory}" 2 72 12 20260813 0
done < "${TOP_FOR_OMP}"

# Repeat the best three settings with a second seed and reversed candidate order.
TOP_THREE="${RUN_ROOT}/control/top_three.tsv"
REPEAT_THREE="${RUN_ROOT}/control/repeat_three.tsv"
rank_trials coarse,refine,pin,omp 3 1 0 "${TOP_THREE}"
if (( NODE_RANK == 0 )); then
    tac "${TOP_THREE}" > "${REPEAT_THREE}"
    touch "${REPEAT_THREE}.ready"
fi
wait_for_file "${REPEAT_THREE}.ready"
repeat_index=0
while read -r workers prefetch pin_memory omp_threads _score _source; do
    repeat_index=$((repeat_index + 1))
    run_trial repeat "repeat${repeat_index}_w${workers}_pf${prefetch}_pin${pin_memory}_omp${omp_threads}" \
        "${workers}" "${prefetch}" "${pin_memory}" "${omp_threads}" 72 12 20260814 0
done < "${REPEAT_THREE}"

# Only repeated settings are eligible for full 173-batch confirmation.
FINAL_TWO="${RUN_ROOT}/control/final_two.tsv"
rank_trials coarse,refine,pin,omp,repeat 2 2 0 "${FINAL_TWO}"
confirm_index=0
while read -r workers prefetch pin_memory omp_threads _score _source; do
    confirm_index=$((confirm_index + 1))
    run_trial confirm "confirm${confirm_index}_w${workers}_pf${prefetch}_pin${pin_memory}_omp${omp_threads}" \
        "${workers}" "${prefetch}" "${pin_memory}" "${omp_threads}" 173 12 20260815 0
done < "${FINAL_TWO}"

if (( NODE_RANK == 0 )); then
    python "${BENCH_TOOL}" report \
        --run-root "${RUN_ROOT}" \
        --output "${RUN_ROOT}/report.md"
    date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_ROOT}/COMPLETED"
    echo "[benchmark_complete] run_root=${RUN_ROOT}"
fi
