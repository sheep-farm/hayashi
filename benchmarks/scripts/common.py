#!/usr/bin/env python3
"""Shared helpers for Hayashi benchmark scripts."""

import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BENCH_DIR.parent
# Prefer the headless hay-run binary if it exists; fall back to the full hay binary.
HAY_EXE = ROOT_DIR / "target" / "release" / "hay-run"
if not HAY_EXE.exists():
    HAY_EXE = ROOT_DIR / "target" / "release" / "hay"
DATASETS_DIR = BENCH_DIR / "datasets" / "generated"
RESULTS_DIR = BENCH_DIR / "results"
TMP_DIR = BENCH_DIR / ".tmp"
ELAPSED_RE = re.compile(r"^\s*elapsed:\s*([0-9]+(?:\.[0-9]+)?)\s*s\s*$", re.IGNORECASE)


def _read_rss_kb(pid: int) -> int:
    """Read VmRSS from /proc/<pid>/status in KB."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])  # KB
    except Exception:
        pass
    return 0


def _find_child_pids(ppid: int) -> list[int]:
    """Return PIDs whose PPid is ppid."""
    children = []
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(entry.name)
            if pid == ppid:
                continue
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("PPid:"):
                        if int(line.split()[1]) == ppid:
                            children.append(pid)
                        break
        except Exception:
            continue
    return children


def _sample_memory_kb(pid: int) -> int:
    """Sum RSS of the main process plus children."""
    total = _read_rss_kb(pid)
    for child in _find_child_pids(pid):
        total += _read_rss_kb(child)
    return total


def parse_elapsed(stdout: bytes) -> list[float]:
    """Extract elapsed times (in seconds) from stdout."""
    times = []
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        m = ELAPSED_RE.match(line)
        if m:
            times.append(float(m.group(1)))
    return times


def measure_run(cmd: list[str]) -> tuple[list[float], int, float]:
    """Run cmd once, return (elapsed_times, peak_memory_kb, wall_time)."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    peak_kb = 0
    t0 = time.perf_counter()
    while proc.poll() is None:
        peak_kb = max(peak_kb, _sample_memory_kb(proc.pid))
        time.sleep(0.01)
    stdout, stderr = proc.communicate()
    wall_time = time.perf_counter() - t0
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, stdout, stderr
        )
    times = parse_elapsed(stdout)
    return times, peak_kb, wall_time


def aggregate(
    times: list[float],
    memory_peaks: list[int],
    wall_times: list[float],
    iters: int,
    warmup: int,
    runs: int,
) -> dict:
    """Aggregate per-call times across runs."""
    if len(times) < 2:
        std = 0.0
    else:
        std = statistics.stdev(times)
    return {
        "mean": statistics.mean(times) if times else 0.0,
        "std": std,
        "min": min(times) if times else 0.0,
        "max": max(times) if times else 0.0,
        "wall_time": statistics.mean(wall_times) if wall_times else 0.0,
        "memory_kb_mean": statistics.mean(memory_peaks) if memory_peaks else 0,
        "memory_kb_max": max(memory_peaks) if memory_peaks else 0,
        "memory_kb_std": statistics.stdev(memory_peaks)
        if len(memory_peaks) > 1
        else 0.0,
        "iters": iters,
        "warmup": warmup,
        "runs": runs,
        "samples": len(times),
    }


def save_results(results: list[dict], name: str) -> Path:
    """Save benchmark results to a timestamped JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{name}_{timestamp}.json"
    path.write_text(json.dumps(results, indent=2))
    return path


def generate_datasets(estimator: str, sizes: list[int]) -> list[Path]:
    """Run the estimator dataset generator and return produced paths."""
    gen = BENCH_DIR / "datasets" / "generate.py"
    sizes_str = ",".join(str(s) for s in sizes)
    subprocess.run(
        [sys.executable, str(gen), "--estimator", estimator, "--sizes", sizes_str],
        check=True,
    )
    return [DATASETS_DIR / f"{estimator}_n{n}.csv" for n in sizes]
