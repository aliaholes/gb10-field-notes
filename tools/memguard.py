#!/usr/bin/env python3
"""memguard.py — run a heavy job on a GB10 / DGX Spark WITHOUT letting it OOM the box.

WHY A WATCHDOG, NOT A CGROUP CAP: measured Oct 2 2026 on a GB10 (unified memory):
  * `systemd-run --user --scope -p MemoryMax=1G` DOES kill a 3 GB CPU allocation (rc 137), but
  * a 3 GB CUDA allocation sailed through a 1.5 GB MemoryMax (driver memory is not charged to the cgroup), while
  * the same CUDA allocation IS visible in /proc/meminfo: 6 GB on cuda dropped MemAvailable by 6.2 GB.
So the only honest wall is a watcher on MemAvailable. The kernel OOM killer may not attribute driver-pinned memory to
the job either, so it could pick the wrong victim (your shell, your editor, a service) — the watchdog acts first.
It cannot see a runtime reclaim stall (MemFree near zero while MemAvailable looks fine): see README section 1.

  memguard.py --floor-gb 16 [--interval 0.5] [--trip-cmd "docker kill my-llm"] [--log F] -- <cmd> [args...]

  * the job runs in its OWN process group with oom_score_adj=1000 (it dies first if the kernel ever chooses)
  * every --interval s: MemAvailable < floor -> run --trip-cmd (if any), SIGTERM the group, 3 s later SIGKILL; exit 99
  * one line per trip + the lowest MemAvailable seen, to stderr (and --log)
  * exit = the job's own exit code when no trip; 99 on a trip; 2 on bad usage
  * --selftest: a known-bad (a CUDA/CPU allocation that crosses the floor MUST trip) + a control (one that doesn't MUST pass)
    CUDA mode needs a python with torch: MEMGUARD_TORCH_PY=/path/to/venv/bin/python (else a CPU bytearray is used)
"""
import os
import signal
import subprocess
import sys
import time


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576.0
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def _child_setup():
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError as e:
        print(f"memguard: could not set oom_score_adj: {e!r}", file=sys.stderr)


def run(cmd, floor_gb, interval=0.5, trip_cmd=None, log=None):
    def say(msg):
        line = f"memguard {time.strftime('%H:%M:%S')}: {msg}"
        print(line, file=sys.stderr, flush=True)
        if log:
            with open(log, "a") as f:
                f.write(line + "\n")

    start = mem_available_gb()
    if start < floor_gb:
        say(f"REFUSED to start: MemAvailable {start:.1f} GB already < floor {floor_gb} GB")
        return 99
    say(f"start: MemAvailable {start:.1f} GB, floor {floor_gb} GB, cmd {cmd[0]}")
    p = subprocess.Popen(cmd, start_new_session=True, preexec_fn=_child_setup)
    low = start
    while True:
        rc = p.poll()
        if rc is not None:
            say(f"job exited rc={rc}; lowest MemAvailable {low:.1f} GB")
            return rc
        avail = mem_available_gb()
        low = min(low, avail)
        if avail < floor_gb:
            say(f"TRIP: MemAvailable {avail:.1f} GB < floor {floor_gb} GB — killing the job")
            if trip_cmd:
                try:
                    r = subprocess.run(trip_cmd, shell=True, capture_output=True, text=True, timeout=30)
                    say(f"trip-cmd rc={r.returncode}: {(r.stdout + r.stderr).strip()[:200]}")
                except Exception as e:
                    say(f"trip-cmd failed: {e!r}")
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            for _ in range(30):
                if p.poll() is not None:
                    break
                time.sleep(0.1)
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
            say(f"killed; MemAvailable now {mem_available_gb():.1f} GB")
            return 99
        time.sleep(interval)


def selftest():
    py = os.environ.get("MEMGUARD_TORCH_PY", "")
    have_cuda = bool(py) and os.path.exists(py)
    alloc = (f"import torch,time; x=torch.empty(int({{gb}}*1024**3),dtype=torch.uint8,device='cuda'); x.fill_(1); "
             f"torch.cuda.synchronize(); time.sleep(3)") if have_cuda else \
            "import time; b=bytearray(int({gb}*1024**3)); time.sleep(3)"
    exe = py if have_cuda else sys.executable
    ok = True
    now = mem_available_gb()
    # known-bad: floor sits 3 GB under the current MemAvailable; the job takes 6 GB -> MUST trip (rc 99)
    rc = run([exe, "-c", alloc.format(gb=6)], floor_gb=now - 3, interval=0.2)
    print(f"[{'PASS' if rc == 99 else 'FAIL'}] known-bad ({'cuda' if have_cuda else 'cpu'} 6 GB vs floor avail-3): rc={rc}")
    ok &= rc == 99
    time.sleep(2)
    now = mem_available_gb()
    # control: floor 10 GB under; the job takes 1 GB -> MUST pass (rc 0)
    rc = run([exe, "-c", alloc.format(gb=1)], floor_gb=now - 10, interval=0.2)
    print(f"[{'PASS' if rc == 0 else 'FAIL'}] control (1 GB vs floor avail-10): rc={rc}")
    ok &= rc == 0
    # refuse-at-start: floor above what exists -> rc 99 without running
    rc = run(["true"], floor_gb=10_000)
    print(f"[{'PASS' if rc == 99 else 'FAIL'}] refuse-at-start: rc={rc}")
    ok &= rc == 99
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv):
    if argv[:1] == ["--selftest"]:
        return selftest()
    floor, interval, trip, log = None, 0.5, None, None
    i = 0
    while i < len(argv) and argv[i] != "--":
        a = argv[i]
        if a == "--floor-gb":
            floor = float(argv[i + 1]); i += 2
        elif a == "--interval":
            interval = float(argv[i + 1]); i += 2
        elif a == "--trip-cmd":
            trip = argv[i + 1]; i += 2
        elif a == "--log":
            log = argv[i + 1]; i += 2
        else:
            print(f"memguard: unknown flag {a!r}\n" + __doc__, file=sys.stderr)
            return 2
    if floor is None or i >= len(argv) - 1:
        print("usage: memguard.py --floor-gb N [--interval S] [--trip-cmd CMD] [--log F] -- <cmd> ...", file=sys.stderr)
        return 2
    return run(argv[i + 1:], floor, interval, trip, log)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
