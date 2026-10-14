#!/usr/bin/env bash
# spark-health.sh — detect the DGX Spark SILENT DEGRADED-CLOCK state under load.
#
# Field report (x.com/Blackwellboy/status/2090611479653622261, Aug 2026): GPU shows
# ~96% util, P-state P0, NO throttle flags — looks busy and healthy — but SM clock is
# pinned ~799 MHz with ~19.5 W under load. Healthy: ~2.3-2.5 GHz, ~92 W. BF16 compute
# 36.5 vs 91.6 TFLOP/s. Idle looks IDENTICAL in both states — detection requires load.
# Remedy per report: FULL power disconnect ~10 min, then boot. (Ours: unplug at the wall + hold power 5-10 s.)
# Aug 21 2026: caught one of our boxes live (513 MHz/14 W); wall-unplug + cap-drain cleared it.
#
# LOAD SOURCES (--load auto|vllm|burn, default auto):
#   vllm — sends a 96-token completion to localhost:8000 and samples clocks during
#          decode. ZERO extra memory — needed because with the full house resident
#          (vLLM 66GB + whisperx + TTS on 121GB unified) a new CUDA context OOMs
#          (proven Aug 22: burn died on first alloc). Also logs tok/s as a free
#          longitudinal decode-speed trend. Healthy calib: SM 2450 MHz, ~35 W.
#   burn — 8-12s bf16 matmul in a fresh torch process (original mode; use when the
#          server is down or for a remote spark without a model loaded).
#   auto — vllm if localhost:8000 answers, else burn.
#
# VERDICTS: PASS | DEGRADED | PROBE-ERROR (exit 3). PROBE-ERROR means NO load was
# generated (burn OOM/no torch AND/OR server absent) — a broken meter is not a sick
# patient; never let it masquerade as either PASS or DEGRADED.
#
# Usage:
#   tools/spark-health.sh                          # probe this box (auto)
#   tools/spark-health.sh user@host                # probe a remote box over ssh
#   Env: SPARK_HEALTH_PY=/path/to/python-with-torch (burn mode; else python3 on PATH)
#   Flags: --load MODE --min-sm MHZ (1500) --min-watts W (40 burn / 0 vllm: advisory)
#          --burn-secs S (12)
#
# THE CAP FALSE ALARM (fixed Oct 7 2026): with the 2000 MHz clock cap (gpu-clock-cap.service, Sep 6) vLLM
# decode draws 18-22 W at 1989 MHz and ~21.5 tok/s (unchanged) -> the old 25 W vllm floor wrote DEGRADED ~4x/day
# for a month (116 false DEGRADED in 124 capped runs; nothing read the flag). Decode watts OVERLAP the degraded state
# (14-20 W) so they cannot discriminate; the CLOCK does (degraded 513-799 MHz vs capped 1989). vllm mode now
# judges on SM clock only and prints watts as advisory. Burn mode keeps 40 W (capped burn: 1969 MHz / 48 W).
#
# SELF-TEST (a check that cannot fail is not a check): --min-sm 999999 MUST print
# DEGRADED (verified Aug 21 burn-mode, Aug 22 vllm-mode); burn-mode under full-house
# memory MUST print PROBE-ERROR, not DEGRADED (verified Aug 22).

set -euo pipefail
HOST=""; MIN_SM=1500; MIN_W=""; BURN=12; LOADMODE="auto"
while [[ $# -gt 0 ]]; do case "$1" in
  --min-sm) MIN_SM=$2; shift 2;; --min-watts) MIN_W=$2; shift 2;;
  --burn-secs) BURN=$2; shift 2;; --load) LOADMODE=$2; shift 2;;
  *) HOST=$1; shift;;
esac; done

PROBE=$(cat <<'EOS'
BURN=__BURN__
LOADMODE=__LOADMODE__
MODE=""
if [ "$LOADMODE" != "burn" ] && curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null | grep -q '"id"'; then
  MODE="vllm"
  T0=$(date +%s.%N)
  curl -s --max-time 180 http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"$(curl -s --max-time 3 http://localhost:8000/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')"'","messages":[{"role":"user","content":"Write a flowing 200-word description of ocean waves. Prose only."}],"max_tokens":96,"temperature":0}' \
    > /tmp/spark-health-vllm.json 2>/dev/null &
  BPID=$!
  MAX_SM=0; MAX_W=0
  while kill -0 $BPID 2>/dev/null; do
    read -r SM W < <(nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader,nounits | tr -d ',' | awk '{print int($1), int($2)}')
    [ "${SM:-0}" -gt "$MAX_SM" ] && MAX_SM=$SM
    [ "${W:-0}" -gt "$MAX_W" ] && MAX_W=$W
    sleep 0.5
  done
  wait $BPID 2>/dev/null || true
  T1=$(date +%s.%N)
  CT=$(grep -o '"completion_tokens":[0-9]*' /tmp/spark-health-vllm.json 2>/dev/null | grep -o '[0-9]*$' || true)
  TOKS=""
  [ -n "$CT" ] && TOKS=$(awk -v c="$CT" -v a="$T0" -v b="$T1" 'BEGIN{printf "%.1f", c/(b-a)}')
  echo "RESULT mode=vllm sm=${MAX_SM} watts=${MAX_W} tflops=n/a toks=${TOKS:-n/a}"
elif [ "$LOADMODE" = "vllm" ]; then
  echo "RESULT mode=vllm sm=0 watts=0 tflops=n/a toks=n/a"
else
  MODE="burn"
  PY=""
  for c in "${SPARK_HEALTH_PY:-}" python3; do
    if [ -n "$c" ] && command -v "$c" >/dev/null && "$c" -c "import torch" 2>/dev/null; then PY="$c"; break; fi
  done
  [ -z "$PY" ] && { echo "RESULT mode=burn sm=0 watts=0 tflops=n/a toks=n/a"; exit 0; }
  "$PY" - <<PYEOF > /tmp/spark-health-burn.log 2>&1 &
import torch, time
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
b = torch.randn_like(a)
torch.cuda.synchronize()
n = 0; t0 = time.time()
while time.time() - t0 < $BURN:
    for _ in range(20): a @ b
    torch.cuda.synchronize(); n += 20
dt = time.time() - t0
print(f"TFLOPS {2*4096**3*n/dt/1e12:.1f}")
PYEOF
  BPID=$!
  sleep 3   # let clocks ramp
  MAX_SM=0; MAX_W=0
  while kill -0 $BPID 2>/dev/null; do
    read -r SM W < <(nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader,nounits | tr -d ',' | awk '{print int($1), int($2)}')
    [ "${SM:-0}" -gt "$MAX_SM" ] && MAX_SM=$SM
    [ "${W:-0}" -gt "$MAX_W" ] && MAX_W=$W
    sleep 1
  done
  wait $BPID 2>/dev/null || true
  TF=$(grep -oE 'TFLOPS [0-9.]+' /tmp/spark-health-burn.log | awk '{print $2}')
  echo "RESULT mode=burn sm=${MAX_SM} watts=${MAX_W} tflops=${TF:-n/a} toks=n/a"
fi
EOS
)
PROBE=${PROBE//__BURN__/$BURN}
PROBE=${PROBE//__LOADMODE__/$LOADMODE}

if [[ -n "$HOST" ]]; then
  OUT=$(ssh -o BatchMode=yes "$HOST" "bash -s" <<<"$PROBE")
else
  OUT=$(bash -s <<<"$PROBE")
fi

echo "$OUT" | grep -v RESULT || true
LINE=$(echo "$OUT" | grep RESULT || true)
[[ -z "$LINE" ]] && { echo "PROBE-ERROR: no result line (probe crashed)"; exit 3; }
MODE=$(sed -E 's/.*mode=([a-z]+).*/\1/' <<<"$LINE")
SM=$(sed -E 's/.*sm=([0-9]+).*/\1/' <<<"$LINE"); W=$(sed -E 's/.*watts=([0-9]+).*/\1/' <<<"$LINE")
TF=$(sed -E 's/.*tflops=([^ ]+).*/\1/' <<<"$LINE"); TOKS=$(sed -E 's/.*toks=([^ ]+).*/\1/' <<<"$LINE")
TARGET=${HOST:-$(hostname)}

# No load generated at all -> broken meter, not a sick patient.
if [[ "$TF" == "n/a" && "$TOKS" == "n/a" ]]; then
  echo "[$TARGET] PROBE-ERROR: no load was generated (mode=$MODE). Burn log: /tmp/spark-health-burn.log; is the model server up?"
  exit 3
fi

[[ -z "$MIN_W" ]] && { [[ "$MODE" == "vllm" ]] && MIN_W=0 || MIN_W=40; }
echo "[$TARGET] under load ($MODE): SM ${SM} MHz | ${W} W | tflops=${TF} | toks/s=${TOKS} (healthy ref: burn ~2300 MHz/~92 W uncapped, 1969 MHz/48 W capped; vllm decode ~2450 MHz/~35 W uncapped, 1989 MHz/18-22 W capped, tok/s unchanged; degraded ref: ~513-799 MHz, ~14-20 W)"
if [[ "$SM" -ge "$MIN_SM" && "$W" -ge "$MIN_W" ]]; then
  echo "[$TARGET] PASS — clocks and power ramp normally under load"
else
  echo "[$TARGET] DEGRADED — SM<${MIN_SM}MHz or power<${MIN_W}W under real load."
  echo "  Remedy (field report + Aug 22 confirmation): full power DISCONNECT at wall, hold power button 5-10s, then boot. Reboot alone does NOT clear it."
  exit 1
fi
