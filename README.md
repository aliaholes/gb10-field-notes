# GB10 field notes (DGX Spark and its siblings)

What we learned running two GB10 boxes (128 GB unified memory, Ubuntu 24.04 / DGX OS) as a 24/7 home lab:
a resident vLLM server, five CUDA audio services, image and video jobs, model quantization. Every note has the
date it was measured, and most say how. **"Measured"** means on our boxes. **"Reported"** means someone else's numbers that we
read but did not reproduce. Where a fix is not confirmed yet, it says so.

Written by Alia Holes, with Kurt Holes. October 2026. Corrections welcome (open an issue).

Contents: [1 Memory](#1-memory-cuda-sees-memfree-not-memavailable) · [2 apt restarts your CUDA services](#2-apt-restarts-your-cuda-services) ·
[3 The silent degraded clock](#3-the-silent-degraded-clock) · [4 vLLM on GB10](#4-vllm-on-gb10) · [5 NVFP4 on GB10](#5-nvfp4-on-gb10) ·
[Tools in this repo](#tools-in-this-repo)

---

## 1. Memory: CUDA sees MemFree, not MemAvailable

**The one fact that explains most GB10 OOMs.** On unified memory, what `cudaMalloc` can get tracks the kernel's
`MemFree`. Page cache, which `free` counts as "available", is not usable by CUDA until the kernel reclaims it, and
it does not reclaim it fast enough for a model load.

- Measured (Sep 1 2026): after a vLLM restart, `MemFree` sat at 11-19 GB behind 33 GB of page cache.
  `free -g` said 41 GB available. `torch.cuda.mem_get_info()` said 19 GB. A speech-to-text service that needs ~12 GB
  OOM'd at `.cuda()` on every 30-second retry for 15 minutes. After `sync; echo 3 > /proc/sys/vm/drop_caches`:
  MemFree 43 GB, CUDA free 42.7 GB, the service loaded in 12 s.
- **Measure with `torch.cuda.mem_get_info()` or `MemFree` in `/proc/meminfo`.** Not the "available" column.

**Fix: drop page cache right before each CUDA service starts.** A root pre-start hook in the unit:

```ini
ExecStartPre=+/usr/local/sbin/free-pages.sh
```

It drops caches only when `MemFree` is below a threshold (page cache is re-readable from disk, so this is safe).
**The threshold must sit above the strictest consumer's own MemFree gate, with margin.** Paid once (Oct 3): the hook's
threshold was 19.07 GiB, a service gate needed 20 GiB, the box had 19.5 GiB free. The hook said "ok" and dropped
nothing; the gate refused the start. We now use 24 GiB. Script: [`tools/free-pages.sh`](tools/free-pages.sh).

**cgroup memory limits do not cap CUDA.** Measured (Oct 2):
- `systemd-run --user --scope -p MemoryMax=1G` killed a 3 GB CPU allocation (exit 137), as expected.
- A 3 GB **CUDA** allocation went straight through a 1.5 GB `MemoryMax`. Driver memory is not charged to the cgroup.
- A separate 6 GB CUDA allocation **did** lower `MemAvailable` by 6.2 GB, so `/proc/meminfo` sees it. A running STT service held 14.8 GB of GPU
  memory while its memcg showed 396 MB (Sep 24).

So the only working wall is a userspace watchdog on `/proc/meminfo`. The kernel OOM killer may also not attribute
driver-pinned memory to the process holding it, and can pick the wrong victim. Our watchdog,
[`tools/memguard.py`](tools/memguard.py), runs a job in its own process group with `oom_score_adj=1000`, polls
`MemAvailable` every 0.5 s, and kills the group below a floor. It has a selftest with a CUDA known-bad.

**But a MemAvailable watchdog cannot see the runtime stall.** Measured (Oct 4): a running vLLM engine froze for
20 minutes (no log lines, 7 requests stuck) while `MemAvailable` read ~31 GB. `MemFree` was 1.7 GB, and `sar -B` showed
direct reclaim (`pgscand` ~2000/s, page faults ×100). The kernel's reserve, `vm.min_free_kbytes`, was ~45 MB. To
diagnose one of these after the fact: `sar -r` and `sar -B` over the window, plus the gap in `docker logs -t`
(timestamps are UTC).

- What we applied: `vm.min_free_kbytes = 4194304` (4 GiB) in `/etc/sysctl.d/`. MemFree went from 2.2 to 8.2 GB.
- **Not confirmed effective yet.** We have not recorded a clean night of `pgscand ≈ 0` since. Treat it as a lead.
- **Side effect, paid the same day:** `MemAvailable` reads ~11 GB lower after the change. Every gate on the box that
  compares against `MemAvailable` trips earlier (one service sat gated for hours). **If you change the kernel reserve,
  re-measure every MemAvailable threshold at the same time.**

**A "background" job that reads a big file into RAM is a heavy job.** Measured (Oct 6): a checksum step read a 22 GB
file whole (`open().read()`, 23.9 GB RSS) beside a GPU benchmark. The kernel logged `NVRM: out of memory`, the CUDA
context was lost, and the benchmark rendered a black frame. Host RSS and the GPU draw from the same pool. Fix: hash in
fixed-size chunks. Re-run: zero NVRM lines.

**Smaller rules we keep:**
- Never start a CUDA service while a vLLM engine on the same box is starting. Its startup memory profiling is the
  worst moment. Wait until its `/metrics` answers. On a full restore: vLLM first, drop caches immediately before
  each CUDA start, start the rest one at a time with a health check between.
- Budget start gates for the load **peak**, not the steady cost. Measured (Oct 3, 1 s sampler): a resident 9B vLLM
  cost 17.3 GiB of MemAvailable at steady state and 23.8 GiB at its load peak, ~6.5 GiB above steady, for ~16 s.
- Memory fit is not compute fit. Two unrelated heavy jobs that fit in memory together still slow each other badly
  (one observation: a training job went from 5 to 16 s/step beside an audio job). We serialize unrelated heavy jobs.
- After a big build or quantize, drop the page cache it left (`posix_fadvise(DONTNEED)` on the files you read, or the
  hook above). Measured: a bf16 model load for quantization took MemFree down to 8.1 GiB while MemAvailable showed 63.7 GB.

---

## 2. apt restarts your CUDA services

On Ubuntu 24.04 the `needrestart` APT hook defaults to **automatic** restart mode. A plain `apt-get install -y`
restarts every daemon linked against an upgraded library, without asking.

Measured (Sep 24): one small install pulled in a security point-upgrade of zlib. needrestart restarted 22 units in the
same few seconds, including all five of our CUDA audio services, plus ssh, rsyslog, polkit and more. They all
cold-started at once into a box full of page cache. The STT service died with `CUDA error: out of memory` and came back
only on a systemd retry after its pre-start hook dropped the cache.

**The procedure we use:**

```bash
apt-get -s install <pkgs>                      # 1. simulate: read what it will touch
sudo NEEDRESTART_MODE=l apt-get install <pkgs> # 2. "l" = list only; nothing is restarted
# 3. restart what it listed BY HAND: drop caches, then ONE CUDA service at a time, health check before the next
```

It worked on its next real use: a later install listed two services as deferred and bounced nothing.

A permanent option we have not adopted: needrestart's `override_rc`. NVIDIA's own DGX OS packages already use it to
protect their services. A file in `/etc/needrestart/conf.d/` with `$nrconf{override_rc}{qr(^your-unit\.service$)} = 0;`
excludes a unit. Untested by us for our own units.

---

## 3. The silent degraded clock

**What it is (reported, then measured on one of our boxes):** under load the GPU shows ~96% utilization, P-state
P0 and **no throttle flags**, but the SM clock stays at ~513-799 MHz and power at ~14-20 W. Usable compute drops to
~40% in the original field report (36.5 vs 91.6 TFLOP/s bf16) and to ~25% on our box (22 vs 88). **At idle it looks normal**, so you only see it under load.

- Measured (Aug 21): the first time our detector ran, it caught one box at **513 MHz / 14 W / 22 TFLOP/s** bf16
  (41 days uptime). The other box ran the same burn at **2242 MHz / 92 W / 88 TFLOP/s**.
- **Fix that worked for us** (the field report waited ~10 min unplugged; draining the button worked for us): shut down, unplug the power brick **at the wall**, hold the power button 5-10 s, plug
  back in, boot. **A reboot does not clear it.** After: 2437 MHz / 82 W, and the following burns 2288-2320 MHz /
  92-93 W / 84.9-88.3 TFLOP/s.
- Likely cause (community reading, not confirmed by NVIDIA): USB-PD power negotiation stuck in a safety contract.
  Reportedly not fixed by a May PD firmware; some units reportedly degrade again daily. To tell a bad brick from a bad
  unit, swap bricks between two units and see whether the fault follows the brick. We moved our brick off a warm
  surface; no burn probe has shown a low clock since.

**Detecting it:** [`tools/spark-health.sh`](tools/spark-health.sh) puts real load on the GPU and reads clock and
power during it. Two load modes:
- `burn`: a 12 s bf16 matmul, reports TFLOP/s. Needs free memory for a new CUDA context.
- `vllm`: times a real request to an OpenAI-compatible server already running on `localhost:8000`. **Zero extra GPU memory**, which matters on a
  full box (our burn mode OOM'd on its first allocation there), and you get tok/s as a free trend line.
- Three verdicts: `PASS`, `DEGRADED`, `PROBE-ERROR`. `PROBE-ERROR` means no load was generated: a broken meter is
  never reported as a healthy or a sick box.

**The false alarm we shipped and then fixed (Oct 7).** In September we capped the GPU clock at 2000 MHz to run cooler
(below). After that, our `vllm`-mode probe reported `DEGRADED` about four times a day for a month: 116 false alarms in 124 runs (the other 8 passed).
Capped decode draws 18-22 W, under the old 25 W floor, while the clock (1989 MHz) and decode speed (~21.5 tok/s,
unchanged) were fine. **Decode watts overlap the degraded state (14-20 W), so they cannot tell the two apart. The
clock can (513-799 vs 1989 MHz).** `vllm` mode now judges on the clock and prints watts as advisory. Burn mode keeps
its 40 W floor (capped burn: 1969 MHz / 48 W). Lesson: a health check written before a config change has to be
re-proven after it. Nothing was reading our flag file, so nobody noticed for a month.

**Clock cap:** `nvidia-smi -lgc 0,2000` at boot (a systemd oneshot, [`tools/gpu-clock-cap.service`](tools/gpu-clock-cap.service);
revert with `nvidia-smi -rgc`). Measured (Sep 6) under sustained render load: before, 83-86 °C, 2203-2411 MHz,
70-80 W; seconds after, 75 °C, 1976 MHz, 46 W.
- **What it costs depends on the workload.** Memory-bound LLM decode: no visible loss (~21.5 tok/s before and after).
  Compute-bound bf16 matmul: ~30% lower (85-88 TF uncapped vs 59.6 TF capped; the capped number is one sample).
- The cap does not show up as a throttle reason (`clocks_event_reasons` stays `0x0`).
- Reported by others: sustained ≥85 °C preceding log-less thermal shutdowns, hotspot spikes to 90-98 °C in seconds,
  no software throttle below 104.8 °C. Linux on our boxes sees no fan tachometer and `-pl` returns N/A. Running out
  of unified memory reportedly gives the same log-less hard power-off, so a silent power-off is not proof of heat.

**Side note:** PassMark lists the GB10 CPU as "ARM Cortex-A725 10 Core 2808 MHz". `lscpu` shows 20 cores, no SMT:
10 Cortex-X925 up to 3900 MHz plus 10 Cortex-A725 up to 2808 MHz. The listing's name and core count describe only
the small cluster.

---

## 4. vLLM on GB10

**Two free flags: +88% batched throughput, no weight change.** Measured (Oct 4) on one ~9B Qwen3.5-family hybrid
(Gated DeltaNet + attention) model in NVFP4 W4A16, vLLM 0.27.2rc0 (torch arch list ending at sm_120, so Marlin W4A16), mean of 3 runs (spread ≤1%):

| | c1 decode tok/s | c16 aggregate tok/s | c16 TTFT p90 |
|---|---|---|---|
| `--enforce-eager`, no prefix cache (base) | 38.58 | 239.3 | 14.00 s |
| CUDA graphs only (drop `--enforce-eager`) | 45.71 (+18%) | 280.1 (+17%) | 11.13 s |
| prefix caching only (`--enable-prefix-caching`) | 38.23 | 387.4 (+62%) | 3.08 s |
| **both** | **45.55 (+18%)** | **448.8 (+88%)** | **2.55 s (−82%)** |

- Graphs mostly help single-stream decode. Prefix caching mostly helps batched throughput and time to first token: our
  requests share a ~2.4k-token system prompt, and its prefill was stealing decode time from every stream.
- Cost: graph capture took 0.51 GiB and 2 s; total startup grew ~29 s (torch.compile + capture).
- **The "`--enforce-eager --no-enable-prefix-caching`" advice in many hybrid-model recipes is out of date** for this
  vLLM. A source read of 0.27.2rc0 found hybrid models explicitly supported for prefix caching, with every disable path
  a graceful auto-off and a log line. On hybrid models vLLM sets Mamba cache mode `align` itself, so caching is coarse
  (1056-token blocks). On a 27B NVFP4 model with multi-token prediction and a ~44k-token shared prefix, prefix
  caching cut warm TTFT 10.8× (61.8 s → 5.7 s), with MTP and vision still working.
- One model only. Whether +88% carries to dense models or other vLLM versions is untested.

**`--kv-cache-memory` skips profiling, but `--gpu-memory-utilization` is still a startup gate.** On unified memory,
pinning the KV pool with `--kv-cache-memory` makes vLLM skip memory profiling. It still refuses to start unless
free memory ≥ utilization × total (~121.6 GiB). Omit `--gpu-memory-utilization` and the gate falls back to 0.92, which
needs ~112 GiB free and fails whenever anything else is resident. **Pin the KV pool and set the utilization
explicitly and low** (by arithmetic, 0.15 × 121.6 GiB → ~18 GiB must be free; the 0.92 failure is the one we hit).

**vLLM's startup memory report undercounts.** For a 27B NVFP4 model with a bf16 vision tower, ~11.5 GiB sat outside
vLLM's own accounting (vision tower, CUDA context, workspace). Plan capacity from `/proc/meminfo`, not the startup
log. On hybrid attention, changing `--max-model-len` also changes the per-token KV cost (we saw ~36.6 KiB/token at
256k vs ~54.8 at 32k), so old ratios don't carry over.

**Greedy decoding is not reproducible on this stack.** At temperature 0: a fresh server vs another fresh server at
concurrency 1 gave 0/16 identical outputs (median first difference at token 64). At concurrency 16, batch composition
changes outputs too (0/50 exact vs a reordered rerun). So **exact-match parity cannot gate a flag or quant change.**
We use teacher-forced token agreement and log-prob drift against the reference instead, plus a noise floor taken from
the reference against a rerun of itself.

**Small ones:**
- Serving a vision-language checkpoint for text only: `--language-model-only` skips the vision tower (our quantized
  vision layers otherwise failed to load, see 5).
- `--max-logprobs 0` stops clients from asking for prompt logprobs. A 10k-token `prompt_logprobs` request produced a
  39,644 MiB (~38.7 GiB) engine high-water mark in our benchmarks.
- If the image's ENTRYPOINT is `vllm serve`, `docker inspect` shows `serve` in Args. Copy that into a new
  `docker run` and it crash-loops with `unrecognized arguments: serve`.
- A checkpoint that ships `kv_cache_quant_algo: FP8` needs `--kv-cache-dtype fp8`. Without it, our 27B gave a wrong,
  deterministic answer to a fixed arithmetic canary (19×23: it answered 417 instead of 437).

---

## 5. NVFP4 on GB10

**The chip has FP4 tensor cores (reported: NVIDIA's docs and vLLM maintainers; we have not run a native FP4 kernel
ourselves).** GB10 is SM121 (`sm_121a` for the arch-specific features). vLLM's log line *"Your GPU does not have native
support for FP4 computation"* describes the **build** (measured on our box: torch 2.11.0+cu130, arch list ending at
sm_120), not the hardware. We got this wrong in our own model cards once, and corrected it. Reportedly, sm_120/120f
builds produce identical cubins for sm_121a, so a dedicated 12.1a build matters mainly for CuTe-DSL / FlashInfer JIT paths.

- On our build, ModelOpt NVFP4 **W4A16** checkpoints load through `MarlinNvFp4LinearKernel`: weight-only, dequantized
  to bf16. Every speed number below is from that path. We have not measured native W4A4 ourselves.
- **Single-stream decode is bandwidth-bound, so native FP4 compute helps it little.** Our 4-bit 9B reads ~4.7 GB of
  weights per token (the 2 GB embedding is a row lookup). At the 273 GB/s spec bandwidth that's a ~57 tok/s ceiling;
  we measured 38.8 (~68%). The 273 is spec, not measured. Reported community numbers for a dense 8B at batch 1
  (vLLM 0.20.2, so possibly affected by the selector bug below): native W4A4 38.6 tok/s vs Marlin W4A16 40.9. FP4 compute pays off in prefill and large batches.
- Reported: a vLLM kernel-selector bug picked the W4A16 CuTe kernel over the W4A4 CUTLASS one for W4A4 checkpoints
  on GB10 (−31% prefill, −17 to −24% decode under load). Fixed in vLLM PR #55170 (merged Sep 8 2026), reportedly not
  in a release through v0.29.0. Workaround: `VLLM_DISABLED_KERNELS=FlashInferCuteDslNvFp4W4A16LinearKernel`. Old
  "W4A4 is slow on GB10" numbers may come from this.

**MLP-only vs all-Linear, measured (Oct 2-3).** A 9.41B hybrid prompt-rewriter, NVFP4 W4A16, same image and flags
for every arm, one GB10:

| | bf16 | MLP-only (96 linears) | all-Linear "MAX" (359) |
|---|---|---|---|
| size | 18.82 GB | 11.87 GB | 6.76 GB |
| teacher-forced top-1 agreement with bf16 (65,824 positions; ceiling 98.83%) | — | 93.54% | 89.88% |
| decode c1 / c16 tok/s (eager, no prefix cache) | 11.8 / 144.5 | 17.8 / 188.2 | 36.6 / 239.8 |
| blind image test vs bf16 (rewrites rendered at the same seed, key sealed) | — | +0.083 ★ | −0.083 ★ |

MAX missed its preregistered 90% agreement bar by 0.12 points and passed the blind test. Both are public:
`kurtholes/PE-T2I-9B-NVFP4-W4A16` and `-MAX` on Hugging Face. A second model (Music Flamingo 8B, all-linear NVFP4) was
2.3× smaller and 3.2× faster, kept lyric accuracy, but drifted and looped more on long free-text answers. Narrowing it to
MLP-only cut the drift only a little and gave back 23% of the speed.

**ModelOpt 0.43 gotchas we hit:**
- `nn.Embedding` can't be quantized: there is no registry entry, and `mtq.quantize` **silently attaches nothing**.
  vLLM 0.27 can't load a packed embedding either. `lm_head` does quantize and load. The embedding stays bf16.
- `max` calibration with a forward loop only sets amax for modules the forward reaches. A text-only calibration never
  reaches a vision tower (110 of 359 quantizers had no amax). Fix: `mtq.calibrate(model, "max")` with no forward loop
  (weight-only), then refuse to export unless every quantizer has a finite amax > 0.
- The export labels layers NVFP4 and writes a W4A4-shaped config group. For W4A16 we rewrite it by hand
  (`quant_algo: W4A16_NVFP4`, group 16, targets = the exact module list) and keep the source `config.json` verbatim.
- The stock `NVFP4_MLP_WEIGHT_ONLY_CFG` uses block 32, and `*mlp*` also matches vision-tower MLPs. Use an explicit dict.
- **Fused layers must share one global scale.** vLLM keeps one `weight_scale_2` per fused layer (`gate_up_proj`,
  `qkv_proj`, ...) and takes the max across the group, so unequal values silently mis-scale blocks. Check
  `weight_scale_2 == amax / 2688` (6 × 448) of the group max. Our exports were unified (88/88 groups).
- On our image, Marlin FP4 refused shapes failing `out % 64 == 0` / `in % 128 == 0` (a separate read of the code found a
  padding path, so this is unresolved). A Qwen vision MLP with intermediate size 4304 was refused, so a
  fully quantized multimodal checkpoint needs `--language-model-only` or bf16 vision layers.
- Per-module NVFP4 dequant error was a steady ~9.4-9.5% relative Frobenius error across layer types. Blocks with
  near-zero norms can underflow the e4m3 scale and dequantize to exactly zero (720 blocks in one vision layer).

---

## Tools in this repo

| file | what | root? | proof |
|---|---|---|---|
| [`tools/memguard.py`](tools/memguard.py) | run a heavy job under a MemAvailable watchdog (own process group, oom_score_adj 1000, kill below a floor) | no | `--selftest` (known-bad must trip, control must pass; CUDA mode needs `MEMGUARD_TORCH_PY`, else CPU) |
| [`tools/spark-health.sh`](tools/spark-health.sh) | detect the silent degraded clock under real load (`burn` or `vllm` mode) | no | `--min-sm 999999` must print DEGRADED |
| [`tools/free-pages.sh`](tools/free-pages.sh) | `ExecStartPre=+` hook: drop page cache when MemFree is below a threshold (`FREE_PAGES_THRESH_KB`) | yes | — |
| [`tools/gpu-clock-cap.service`](tools/gpu-clock-cap.service) | cap the GPU clock at boot | yes | revert: `nvidia-smi -rgc`. Fixed Oct 7: an ordering cycle that could drop it at boot |

All MIT. Read them before running them as root. They were written for our boxes; paths and thresholds are flags or env vars. Dropping page cache costs some disk re-reads, nothing else.
