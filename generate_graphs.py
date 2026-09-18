#!/usr/bin/env python3
"""
generate_graphs.py
==================
Reads results/benchmark.csv produced by the CUDA DNA alignment program
and generates two publication-quality performance graphs:

  1. CPU vs GPU Execution Time vs Sequence Size
  2. Speedup (Kernel & Total) vs Sequence Size

Usage:
    python generate_graphs.py [--csv PATH] [--out DIR]

Requires:
    pip install matplotlib pandas
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # Non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "benchmark.csv")
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "results")

# ── Color palette ─────────────────────────────────────────────────────────────
COLOR_CPU    = "#E74C3C"   # red
COLOR_KERNEL = "#2ECC71"   # green
COLOR_TOTAL  = "#3498DB"   # blue
COLOR_SPEED_K = "#9B59B6"  # purple
COLOR_SPEED_T = "#F39C12"  # orange


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        sys.exit(f"[ERROR] CSV not found: {csv_path}\n"
                 "Run the CUDA program first to generate results/benchmark.csv")
    df = pd.read_csv(csv_path)
    required = {"SeqLen", "CPU_ms", "GPUKernel_ms", "GPUTotal_ms",
                "SpeedupKernel", "SpeedupTotal"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] CSV missing columns: {missing}")
    # If the same sequence length was run multiple times, average them
    df = df.groupby("SeqLen", as_index=False).mean(numeric_only=True)
    df = df.sort_values("SeqLen")
    return df


def plot_time_comparison(df: pd.DataFrame, out_dir: str):
    """Plot CPU Time vs GPU Kernel Time vs GPU Total Time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    ax.plot(df["SeqLen"], df["CPU_ms"],       "o-", color=COLOR_CPU,
            linewidth=2.5, markersize=7, label="CPU Time")
    ax.plot(df["SeqLen"], df["GPUKernel_ms"], "s-", color=COLOR_KERNEL,
            linewidth=2.5, markersize=7, label="GPU Kernel Time")
    ax.plot(df["SeqLen"], df["GPUTotal_ms"],  "^-", color=COLOR_TOTAL,
            linewidth=2.5, markersize=7, label="GPU Total Time (incl. mem)")

    ax.set_xlabel("Sequence Length", fontsize=13, color="white")
    ax.set_ylabel("Execution Time (ms)", fontsize=13, color="white")
    ax.set_title("CPU vs CUDA Execution Time\n(Needleman-Wunsch DNA Alignment)",
                 fontsize=15, color="white", fontweight="bold")
    ax.legend(fontsize=11, facecolor="#0F3460", edgecolor="white",
              labelcolor="white")
    ax.tick_params(colors="white")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.grid(True, color="#2A2A4A", linewidth=0.8)

    out_path = os.path.join(out_dir, "time_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"[Graph] Saved: {out_path}")
    plt.close()


def plot_speedup(df: pd.DataFrame, out_dir: str):
    """Plot GPU speedup (kernel-only and total) vs sequence length."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    ax.plot(df["SeqLen"], df["SpeedupKernel"], "o-", color=COLOR_SPEED_K,
            linewidth=2.5, markersize=7, label="Speedup (Kernel only)")
    ax.plot(df["SeqLen"], df["SpeedupTotal"],  "s-", color=COLOR_SPEED_T,
            linewidth=2.5, markersize=7, label="Speedup (Total incl. mem)")
    ax.axhline(y=1.0, color="white", linewidth=1, linestyle="--",
               alpha=0.5, label="Baseline (1×)")

    ax.set_xlabel("Sequence Length", fontsize=13, color="white")
    ax.set_ylabel("Speedup (×)", fontsize=13, color="white")
    ax.set_title("CUDA Speedup over CPU\n(Needleman-Wunsch DNA Alignment)",
                 fontsize=15, color="white", fontweight="bold")
    ax.legend(fontsize=11, facecolor="#0F3460", edgecolor="white",
              labelcolor="white")
    ax.tick_params(colors="white")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.grid(True, color="#2A2A4A", linewidth=0.8)

    out_path = os.path.join(out_dir, "speedup.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"[Graph] Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate performance graphs from benchmark CSV")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to benchmark.csv")
    parser.add_argument("--out", default=DEFAULT_OUT,  help="Output directory for PNG files")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = load_data(args.csv)
    print(f"[Data] Loaded {len(df)} rows from {args.csv}")
    print(df[["SeqLen","CPU_ms","GPUKernel_ms","GPUTotal_ms",
              "SpeedupKernel","SpeedupTotal"]].to_string(index=False))

    plot_time_comparison(df, args.out)
    plot_speedup(df, args.out)
    print("\n[Done] Both graphs saved to:", args.out)


if __name__ == "__main__":
    main()
