#!/usr/bin/env bash
set -euo pipefail

# Parallel test runner for EMLX backend using ExUnit partitions.
# Why: EMLX is unstable with max_cases > 1 in a single VM, but stable
# across multiple partitioned VMs.
#
# Usage:
#   scripts/test_emlx_partitioned.sh
#   scripts/test_emlx_partitioned.sh 4
#   scripts/test_emlx_partitioned.sh 4 -- --exclude external
#   BSTS_NX_TEST_BACKEND=emlx scripts/test_emlx_partitioned.sh 6 -- test/bsts_nx/applications/*.exs
#
# Env vars:
#   BSTS_NX_TEST_BACKEND  Backend to use (default: emlx)
#   BSTS_NX_MAX_CASES     max_cases per partition VM (default: 1)
#   BSTS_NX_LOG_DIR       optional log directory
#   BSTS_NX_SKIP_COMPILE  set to 1 to skip the initial precompile step

PARTITIONS="${1:-4}"
if [[ "${PARTITIONS}" == "--" ]]; then
  PARTITIONS=4
fi

if ! [[ "${PARTITIONS}" =~ ^[0-9]+$ ]] || [[ "${PARTITIONS}" -lt 1 ]]; then
  echo "error: first arg must be a positive integer partition count (got: ${PARTITIONS})" >&2
  exit 2
fi

shift $(( $# > 0 ? 1 : 0 )) || true

if [[ "${1:-}" == "--" ]]; then
  shift
fi

BACKEND="${BSTS_NX_TEST_BACKEND:-emlx}"
MAX_CASES="${BSTS_NX_MAX_CASES:-1}"

if [[ -n "${BSTS_NX_LOG_DIR:-}" ]]; then
  LOG_DIR="${BSTS_NX_LOG_DIR}"
else
  LOG_DIR="/tmp/bsts_nx_test_${BACKEND}_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "${LOG_DIR}"

if [[ "${BSTS_NX_SKIP_COMPILE:-0}" != "1" ]]; then
  echo "==> Precompiling once for backend ${BACKEND}..."
  env BSTS_NX_TEST_BACKEND="${BACKEND}" mix compile >/dev/null
else
  echo "==> Skipping precompile (BSTS_NX_SKIP_COMPILE=1)"
fi

echo "==> Running ${PARTITIONS} partition(s) with backend=${BACKEND}, max_cases=${MAX_CASES}"
echo "==> Logs: ${LOG_DIR}"

PIDS=()

for p in $(seq 1 "${PARTITIONS}"); do
  (
    env \
      BSTS_NX_TEST_BACKEND="${BACKEND}" \
      MIX_TEST_PARTITION="${p}" \
      mix test \
      --partitions "${PARTITIONS}" \
      --max-cases "${MAX_CASES}" \
      --no-compile \
      "$@" >"${LOG_DIR}/partition_${p}.log" 2>&1
  ) &
  PIDS+=("$!")
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done

echo ""
echo "==> Partition summaries"
for p in $(seq 1 "${PARTITIONS}"); do
  LOG_FILE="${LOG_DIR}/partition_${p}.log"
  echo "--- partition ${p} ---"
  grep -E "Running ExUnit|Finished in|[0-9]+ tests, [0-9]+ failures" "${LOG_FILE}" || true
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo ""
  echo "One or more partitions failed. Showing tail of failing logs:"
  for p in $(seq 1 "${PARTITIONS}"); do
    LOG_FILE="${LOG_DIR}/partition_${p}.log"
    if ! grep -q "0 failures" "${LOG_FILE}"; then
      echo "----- tail: partition ${p} -----"
      tail -n 80 "${LOG_FILE}" || true
    fi
  done
  exit 1
fi

echo ""
echo "All partitions passed."
