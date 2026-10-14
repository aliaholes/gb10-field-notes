#!/bin/sh
# free-pages.sh — run as root (ExecStartPre=+) before a CUDA service loads its models.
# GB10 lesson (Sep 1 2026): CUDA's free memory tracks the kernel's MemFree, NOT MemAvailable — page
# cache the kernel calls "available" is not available to cudaMalloc. A speech-to-text model needing ~12 GB of
# TRULY free pages (CPU staging + the .cuda() copy) OOM'd on every retry for 15 minutes after a vLLM restart left
# MemFree at 11-19 GB behind 33 GB of page cache.
# Use in a unit:  ExecStartPre=+/usr/local/sbin/free-pages.sh   (the + runs it as root)
# Dropping page cache is benign (re-readable from disk). Only do it when MemFree is actually low.
# THRESHOLD LESSON (Oct 3 2026): was 20000000 kB (= 19.07 GiB) while a service's own MemFree gate needed 20 GiB
# (20971520 kB): MemFree 19.5 GiB sat BETWEEN them -> "ok", no drop, the gate REFUSED the start. The threshold must sit
# ABOVE the strictest consumer gate with margin: 24 GiB. Dropping page cache is benign, so erring high costs nothing.
THRESH_KB=${FREE_PAGES_THRESH_KB:-25165824}   # 24 GiB default; set it ABOVE your strictest service's need
free_kb=$(awk '/^MemFree:/{print $2}' /proc/meminfo)
if [ "$free_kb" -lt "$THRESH_KB" ]; then
  sync
  echo 3 > /proc/sys/vm/drop_caches
  now_kb=$(awk '/^MemFree:/{print $2}' /proc/meminfo)
  echo "[free-pages] MemFree ${free_kb} kB < ${THRESH_KB} kB -> dropped page cache -> ${now_kb} kB"
else
  echo "[free-pages] MemFree ${free_kb} kB ok"
fi
exit 0
