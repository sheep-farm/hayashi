#!/usr/bin/env python3
"""Gera tabela Markdown e gráficos a partir dos resultados JSON do benchmark."""

import json
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_results() -> list[dict]:
    """Carrega o resultado mais recente por (estimator, n, language)."""
    records = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        mtime = path.stat().st_mtime
        data = json.loads(path.read_text())
        if isinstance(data, list):
            for item in data:
                item["_mtime"] = mtime
                records.append(item)
        else:
            data["_mtime"] = mtime
            records.append(data)

    latest = {}
    for r in records:
        key = (r["estimator"], r["n"], r["language"])
        if key not in latest or r["_mtime"] > latest[key]["_mtime"]:
            latest[key] = r

    return sorted(latest.values(), key=lambda x: (x["estimator"], x["n"], x["language"]))


def build_table(results: list[dict]) -> str:
    lines = [
        "# Benchmark Summary",
        "",
        "| Estimator | n | Language | Mean (s) | Std (s) | Memory (MB) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        mem_mb = r.get("memory_kb_mean", 0) / 1024
        lines.append(
            f"| {r['estimator']} | {r['n']} | {r['language']} | "
            f"{r['mean']:.4f} | {r['std']:.4f} | {mem_mb:.1f} |"
        )
    return "\n".join(lines) + "\n"


def build_speedup_table(results: list[dict]) -> str:
    """Tabela de speedup do Hayashi vs cada linguagem por estimador/n."""
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["estimator"], r["n"])].append(r)

    lines = [
        "",
        "## Speedup Hayashi vs concorrentes",
        "",
        "| Estimator | n | vs Python | vs R |",
        "|---|---|---:|---:|",
    ]
    for (est, n), group in sorted(grouped.items()):
        by_lang = {r["language"]: r for r in group}
        if "hayashi" not in by_lang:
            continue
        hay = by_lang["hayashi"]["mean"]
        py = by_lang.get("python", {}).get("mean")
        r_ = by_lang.get("r", {}).get("mean")
        py_spd = f"{py / hay:.1f}x" if py else "—"
        r_spd = f"{r_ / hay:.1f}x" if r_ else "—"
        lines.append(f"| {est} | {n} | {py_spd} | {r_spd} |")

    return "\n".join(lines) + "\n"


def plot_results(results: list[dict], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot generation")
        return

    grouped = defaultdict(list)
    for r in results:
        grouped[r["estimator"]].append(r)

    n_estimators = len(grouped)
    fig, axes = plt.subplots(1, n_estimators, figsize=(5 * n_estimators, 4), squeeze=False)
    axes = axes[0]

    colors = {"hayashi": "#1f77b4", "python": "#ff7f0e", "r": "#2ca02c"}

    for ax, (est, rows) in zip(axes, sorted(grouped.items())):
        by_lang = defaultdict(list)
        ns = sorted({r["n"] for r in rows})
        for n in ns:
            for r in rows:
                if r["n"] == n:
                    by_lang[r["language"]].append((n, r["mean"]))

        for lang, points in sorted(by_lang.items()):
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", label=lang, color=colors.get(lang))

        ax.set_xlabel("n")
        ax.set_ylabel("mean time (s)")
        ax.set_title(est)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Plot saved to {output}")


def main():
    if not RESULTS_DIR.exists():
        print(f"No results directory at {RESULTS_DIR}")
        sys.exit(1)

    results = load_results()
    if not results:
        print("No result files found.")
        sys.exit(1)

    table = build_table(results)
    speedup = build_speedup_table(results)
    summary_path = RESULTS_DIR / "summary.md"
    summary_path.write_text(table + speedup)
    print(f"Summary written to {summary_path}")

    plot_path = RESULTS_DIR / "summary.png"
    plot_results(results, plot_path)


if __name__ == "__main__":
    main()
