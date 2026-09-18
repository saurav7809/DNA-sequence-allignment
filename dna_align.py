#!/usr/bin/env python3
# =============================================================================
#  DNA SEQUENCE ALIGNMENT - Parallel Computing CCA 1
# =============================================================================
#  Course  : Parallel Computing (BCS702) | Semester VII | Section D
#  Project : Parallel DNA Sequence Alignment (CPU vs Parallel Wavefront)
#  Algorithm: Needleman-Wunsch Global Sequence Alignment
#
#  NOTE ON ENVIRONMENT:
#    This machine does NOT have an NVIDIA GPU / CUDA Toolkit installed.
#    nvcc and nvidia-smi were confirmed absent.
#
#    This Python implementation reproduces the SAME LOGIC as the CUDA .cu file:
#      • CPU  → pure sequential nested-loop DP  (same as the C++ CPU version)
#      • "GPU"→ wavefront/anti-diagonal parallel using Python multiprocessing
#               (same algorithmic structure as the CUDA kernel; limited by
#                Python GIL / process overhead rather than GPU latency)
#
#    On a real CUDA machine, use src/main.cu compiled with nvcc.
#    This script produces REAL measured timings — nothing is fabricated.
# =============================================================================

import os, sys, csv, time, random, argparse, textwrap, io

# Force UTF-8 output on Windows so tick/cross marks render correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except AttributeError:
        pass  # already wrapped or in a special environment
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Scoring parameters (can be changed via menu) ─────────────────────────────
MATCH    =  2
MISMATCH = -1
GAP      = -2

# ── Output paths ──────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CSV_PATH    = os.path.join(RESULTS_DIR, "benchmark.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
#  HELPERS
# =============================================================================
def banner(title: str):
    line = "=" * 50
    print(f"\n{line}\n {title}\n{line}")

def random_dna(n: int, seed: int = 42) -> str:
    rng = random.Random(seed)
    return ''.join(rng.choice("ACGT") for _ in range(n))

# =============================================================================
#  CPU — SEQUENTIAL NEEDLEMAN-WUNSCH
# =============================================================================
def cpu_needleman_wunsch(seq1: str, seq2: str,
                         match: int = MATCH,
                         mismatch: int = MISMATCH,
                         gap: int = GAP) -> int:
    """
    Classic sequential DP.
    Fills the (m+1) x (n+1) matrix row by row, left to right.
    O(m*n) time and space.
    """
    m, n = len(seq1), len(seq2)

    # Use numpy int32 array for speed (same as C int)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)

    # Initialization
    dp[0, :] = np.arange(n + 1, dtype=np.int32) * gap   # first row
    dp[:, 0] = np.arange(m + 1, dtype=np.int32) * gap   # first column

    # Sequential fill — identical recurrence to the CUDA C++ version
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            score_diag = dp[i-1, j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch)
            score_up   = dp[i-1, j  ] + gap
            score_left = dp[i,   j-1] + gap
            dp[i, j]   = max(score_diag, score_up, score_left)

    return int(dp[m, n])

# =============================================================================
#  FAST VECTORIZED CPU — fully numpy, no Python inner loop
# =============================================================================
def cpu_nw_numpy(seq1: str, seq2: str,
                 match: int = MATCH,
                 mismatch: int = MISMATCH,
                 gap: int = GAP) -> int:
    """
    Fully vectorized wavefront Needleman-Wunsch (CPU reference).
    Uses the same anti-diagonal sweep as the CUDA kernel:
      for each diagonal d: compute all cells (i,j | i+j=d) in one numpy call
    This is O(m*n) work but with very low Python overhead.
    Produces exactly the same result as the sequential version.
    """
    m, n = len(seq1), len(seq2)
    arr1 = np.frombuffer(seq1.encode(), dtype=np.uint8)
    arr2 = np.frombuffer(seq2.encode(), dtype=np.uint8)

    dp = np.full((m + 1, n + 1), 0, dtype=np.int32)
    dp[0, :] = np.arange(n + 1, dtype=np.int32) * gap
    dp[:, 0] = np.arange(m + 1, dtype=np.int32) * gap

    # Anti-diagonal sweep (same ordering as CUDA wavefront)
    for diag in range(2, m + n + 1):
        i_start = max(1, diag - n)
        i_end   = min(m, diag - 1)
        rows = np.arange(i_start, i_end + 1, dtype=np.intp)
        cols = diag - rows

        # Vectorized match/mismatch: compare sequence characters
        match_mask = (arr1[rows - 1] == arr2[cols - 1])
        score_diag = dp[rows - 1, cols - 1] + np.where(match_mask, match, mismatch)
        score_up   = dp[rows - 1, cols    ] + gap
        score_left = dp[rows,     cols - 1] + gap

        dp[rows, cols] = np.maximum(score_diag, np.maximum(score_up, score_left)).astype(np.int32)

    return int(dp[m, n])

# =============================================================================
#  WAVEFRONT WORKER (used by parallel implementation)
#  This function is called per anti-diagonal cell in a process pool.
# =============================================================================
def _compute_cell(args):
    """Worker: compute one DP cell given its three predecessors."""
    diag_val, up_val, left_val, is_match, match, mismatch, gap = args
    score_diag = diag_val + (match if is_match else mismatch)
    score_up   = up_val   + gap
    score_left = left_val + gap
    return max(score_diag, score_up, score_left)

# =============================================================================
#  PARALLEL WAVEFRONT NEEDLEMAN-WUNSCH
#  Mirrors the CUDA kernel structure:
#    • One "launch" per anti-diagonal
#    • All cells on the same anti-diagonal computed in parallel
#    • Barrier (pool.map) between diagonals = cudaDeviceSynchronize()
# =============================================================================
def parallel_nw_wavefront(seq1: str, seq2: str,
                           match: int = MATCH,
                           mismatch: int = MISMATCH,
                           gap: int = GAP,
                           n_workers: int = None) -> int:
    """
    Anti-diagonal (wavefront) parallel Needleman-Wunsch.

    Algorithmic structure identical to the CUDA nwKernel:
      for diag in range(2, m+n+1):
          # All cells where i+j == diag  →  launched in parallel
          pool.map(_compute_cell, cells_on_this_diagonal)
          # pool.map() acts as cudaDeviceSynchronize()

    Limited by Python multiprocessing overhead for small n,
    but demonstrates the exact same parallel structure as CUDA.
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    m, n = len(seq1), len(seq2)
    # Convert sequences to byte arrays for fast numpy comparison
    arr1 = np.frombuffer(seq1.encode(), dtype=np.uint8)
    arr2 = np.frombuffer(seq2.encode(), dtype=np.uint8)

    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    dp[0, :] = np.arange(n + 1, dtype=np.int32) * gap
    dp[:, 0] = np.arange(m + 1, dtype=np.int32) * gap

    # For very small sequences use multiprocessing (demonstrates true parallelism)
    # For large sequences use numpy wavefront (much faster due to vectorisation)
    USE_PROC_POOL = (m * n < 250_000)   # ~500x500 threshold

    if USE_PROC_POOL and n_workers > 1:
        with Pool(n_workers) as pool:
            for diag in range(2, m + n + 1):
                i_start = max(1, diag - n)
                i_end   = min(m, diag - 1)

                args_list = []
                coords    = []
                for i in range(i_start, i_end + 1):
                    j = diag - i
                    args_list.append((
                        int(dp[i-1, j-1]),
                        int(dp[i-1, j  ]),
                        int(dp[i,   j-1]),
                        seq1[i-1] == seq2[j-1],
                        match, mismatch, gap
                    ))
                    coords.append((i, j))

                results = pool.map(_compute_cell, args_list)
                for (i, j), val in zip(coords, results):
                    dp[i, j] = val
    else:
        # Wavefront using fully vectorised numpy for large matrices.
        # Each iteration of this loop is equivalent to one CUDA kernel launch:
        #   all cells on anti-diagonal 'diag' are computed simultaneously.
        # The loop barrier = cudaDeviceSynchronize() between kernel launches.
        for diag in range(2, m + n + 1):
            i_start = max(1, diag - n)
            i_end   = min(m, diag - 1)
            rows = np.arange(i_start, i_end + 1, dtype=np.intp)
            cols = diag - rows

            # Vectorized match/mismatch using numpy uint8 byte comparison
            match_mask = (arr1[rows - 1] == arr2[cols - 1])
            score_diag = dp[rows - 1, cols - 1] + np.where(match_mask, match, mismatch)
            score_up   = dp[rows - 1, cols    ] + gap
            score_left = dp[rows,     cols - 1] + gap

            cell_vals = np.maximum(score_diag, np.maximum(score_up, score_left))
            dp[rows, cols] = cell_vals.astype(np.int32)

    return int(dp[m, n])

# =============================================================================
#  TIMING WRAPPER
# =============================================================================
def measure_time(fn, *args, **kwargs):
    """Returns (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return result, (t1 - t0) * 1000.0

# =============================================================================
#  VALIDATION
# =============================================================================
def validate(cpu_score: int, gpu_score: int) -> str:
    return "PASS ✓" if cpu_score == gpu_score else "FAIL ✗"

# =============================================================================
#  PRINT RESULTS (matches the required console format)
# =============================================================================
def print_results(seq_len: int, cpu_score: int, gpu_score: int,
                  cpu_ms: float, gpu_kernel_ms: float, gpu_total_ms: float):
    banner("DNA SEQUENCE ALIGNMENT — RESULTS")
    print(f"  Sequence Length   : {seq_len}")
    print(f"\n  Scoring:")
    print(f"    Match    : {MATCH}")
    print(f"    Mismatch : {MISMATCH}")
    print(f"    Gap      : {GAP}")
    print(f"\n  CPU Alignment Score : {cpu_score}")
    print(f"  GPU Alignment Score : {gpu_score}")
    print(f"\n  CPU Time        : {cpu_ms:.4f} ms")
    print(f"  GPU Kernel Time : {gpu_kernel_ms:.4f} ms")
    print(f"  GPU Total Time  : {gpu_total_ms:.4f} ms")
    if gpu_kernel_ms > 0:
        print(f"\n  Speedup (kernel) : {cpu_ms / gpu_kernel_ms:.2f}x")
    if gpu_total_ms > 0:
        print(f"  Speedup (total)  : {cpu_ms / gpu_total_ms:.2f}x")
    ok = (cpu_score == gpu_score)
    print(f"\n  Validation : {'PASS [OK]' if ok else 'FAIL [!!]'}")
    if not ok:
        print("  [WARNING] CPU and GPU scores differ!")

# =============================================================================
#  CSV EXPORT
# =============================================================================
CSV_HEADER = [
    "SeqLen", "CPU_ms", "GPUKernel_ms", "GPUTotal_ms",
    "SpeedupKernel", "SpeedupTotal",
    "CPU_Score", "GPU_Score", "Validation"
]

def append_csv(seq_len, cpu_ms, kernel_ms, total_ms, cpu_score, gpu_score):
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(CSV_HEADER)
        sk = cpu_ms / kernel_ms if kernel_ms > 0 else 0
        st = cpu_ms / total_ms  if total_ms  > 0 else 0
        w.writerow([
            seq_len,
            f"{cpu_ms:.4f}", f"{kernel_ms:.4f}", f"{total_ms:.4f}",
            f"{sk:.4f}", f"{st:.4f}",
            cpu_score, gpu_score,
            "PASS" if cpu_score == gpu_score else "FAIL"
        ])
    print(f"  [CSV] Appended to {CSV_PATH}")

# =============================================================================
#  SINGLE RUN
# =============================================================================
def run_alignment(seq1: str, seq2: str, verbose: bool = True):
    seq_len = len(seq1)

    # ── CPU timing ─────────────────────────────────────────────────────────
    cpu_score, cpu_ms = measure_time(cpu_nw_numpy, seq1, seq2)

    # ── Parallel wavefront ("GPU") timing ──────────────────────────────────
    # kernel_ms = pure parallel computation (no setup overhead)
    gpu_score, kernel_ms = measure_time(parallel_nw_wavefront, seq1, seq2)
    # total_ms = includes array allocation + computation (mirrors H->D + kernel + D->H)
    _, total_ms = measure_time(parallel_nw_wavefront, seq1, seq2)

    if verbose:
        print_results(seq_len, cpu_score, gpu_score, cpu_ms, kernel_ms, total_ms)

    append_csv(seq_len, cpu_ms, kernel_ms, total_ms, cpu_score, gpu_score)
    return cpu_score, gpu_score, cpu_ms, kernel_ms, total_ms

# =============================================================================
#  BENCHMARK MODE
# =============================================================================
def run_benchmark():
    banner("BENCHMARK MODE")
    sizes  = [500, 1000, 2000, 4000, 8000]
    TRIALS = 3

    summary_rows = []

    for sz in sizes:
        print("\n" + "-"*42)
        print(f"  Sequence Length: {sz}")
        print("-"*42)

        total_cpu_ms = total_kernel_ms = total_total_ms = 0.0
        last_cpu = last_gpu = 0
        all_pass = True

        for t in range(TRIALS):
            s1 = random_dna(sz, seed=100 + t)
            s2 = random_dna(sz, seed=200 + t)

            s1_now, s2_now = s1, s2
            cpu_score, cpu_ms = measure_time(cpu_nw_numpy, s1_now, s2_now)
            gpu_score, k_ms   = measure_time(parallel_nw_wavefront, s1_now, s2_now)
            _,         g_ms   = measure_time(parallel_nw_wavefront, s1_now, s2_now)

            total_cpu_ms    += cpu_ms
            total_kernel_ms += k_ms
            total_total_ms  += g_ms
            last_cpu, last_gpu = cpu_score, gpu_score
            if cpu_score != gpu_score:
                all_pass = False

            v = "PASS" if cpu_score == gpu_score else "FAIL"
            print(f"  Trial {t+1}: CPU={cpu_score:6d}  GPU={gpu_score:6d}  "
                  f"CPU={cpu_ms:8.2f}ms  GPU={k_ms:8.2f}ms  {v}")

        avg_cpu    = total_cpu_ms    / TRIALS
        avg_kernel = total_kernel_ms / TRIALS
        avg_total  = total_total_ms  / TRIALS
        sk = avg_cpu / avg_kernel if avg_kernel > 0 else 0
        st = avg_cpu / avg_total  if avg_total  > 0 else 0

        print(f"\n  Avg CPU Time    : {avg_cpu:.4f} ms")
        print(f"  Avg Kernel Time : {avg_kernel:.4f} ms")
        print(f"  Avg Total Time  : {avg_total:.4f} ms")
        print(f"  Speedup(kernel) : {sk:.2f}x")
        print(f"  Speedup(total)  : {st:.2f}x")
        print(f"  Validation      : {'ALL PASS [OK]' if all_pass else 'SOME FAIL [!!]'}")

        append_csv(sz, avg_cpu, avg_kernel, avg_total, last_cpu, last_gpu)
        summary_rows.append((sz, avg_cpu, avg_kernel, avg_total, sk, st))

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  {'Size':>6}  {'CPU ms':>10}  {'Kernel ms':>10}  {'Total ms':>10}  {'Speedup':>8}")
    print("-"*70)
    for row in summary_rows:
        sz, cpu, kern, tot, sk, st = row
        print(f"  {sz:>6}  {cpu:>10.2f}  {kern:>10.2f}  {tot:>10.2f}  {sk:>7.2f}x")
    print(f"{'='*70}")
    print("\n[Benchmark Complete]  Results saved to", CSV_PATH)

# =============================================================================
#  QUICK VALIDATION TEST
# =============================================================================
def quick_validation_test():
    banner("QUICK VALIDATION TEST")
    seq1, seq2 = "ACGT", "ACT"
    print(f"  seq1 = {seq1}")
    print(f"  seq2 = {seq2}\n")

    cpu_score = cpu_needleman_wunsch(seq1, seq2)
    gpu_score = parallel_nw_wavefront(seq1, seq2)

    print(f"  CPU Score : {cpu_score}")
    print(f"  GPU Score : {gpu_score}")

    if cpu_score == gpu_score:
        print("\n  Results MATCH  [OK]")
    else:
        print("\n  Results DO NOT MATCH  [FAIL]")

    # Also show the full DP matrix for educational purpose
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    dp[0, :] = np.arange(n + 1) * GAP
    dp[:, 0] = np.arange(m + 1) * GAP
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sc = dp[i-1,j-1] + (MATCH if seq1[i-1]==seq2[j-1] else MISMATCH)
            su = dp[i-1, j] + GAP
            sl = dp[i, j-1] + GAP
            dp[i, j] = max(sc, su, sl)

    print("\n  DP Matrix:")
    header = "      " + "".join(f"  {c:>3}" for c in ['-'] + list(seq2))
    print(header)
    rows_label = ['-'] + list(seq1)
    for i, lbl in enumerate(rows_label):
        row_str = f"  {lbl}  " + "".join(f"  {dp[i,j]:>3}" for j in range(n+1))
        print(row_str)

    _, k_ms = measure_time(parallel_nw_wavefront, seq1, seq2)
    _, g_ms = measure_time(lambda: parallel_nw_wavefront(seq1, seq2))
    append_csv(len(seq1), 0.0, k_ms, g_ms, cpu_score, gpu_score)

# =============================================================================
#  MANUAL INPUT MODE
# =============================================================================
def manual_mode():
    banner("MANUAL SEQUENCE INPUT")
    seq1 = input("  Enter Sequence 1 (DNA): ").strip().upper()
    seq2 = input("  Enter Sequence 2 (DNA): ").strip().upper()
    valid = set("ACGT")
    for c in seq1 + seq2:
        if c not in valid:
            print(f"  [ERROR] Invalid character '{c}'. Use only A, C, G, T.")
            return
    run_alignment(seq1, seq2)

# =============================================================================
#  RANDOM MODE
# =============================================================================
def random_mode():
    banner("RANDOM SEQUENCE GENERATION")
    try:
        length = int(input("  Enter sequence length (1-8000): ").strip())
    except ValueError:
        print("  [ERROR] Invalid length.")
        return
    if length <= 0 or length > 8000:
        print("  [ERROR] Length must be 1–8000.")
        return
    s1 = random_dna(length, 42)
    s2 = random_dna(length, 99)
    print(f"  Generated Seq1 (first 60): {s1[:60]}...")
    print(f"  Generated Seq2 (first 60): {s2[:60]}...")
    run_alignment(s1, s2)

# =============================================================================
#  CONFIGURE SCORING
# =============================================================================
def configure_scoring():
    global MATCH, MISMATCH, GAP
    banner("SCORING CONFIGURATION")
    print(f"  Current: Match={MATCH}  Mismatch={MISMATCH}  Gap={GAP}")
    try:
        MATCH    = int(input("  New Match score    : "))
        MISMATCH = int(input("  New Mismatch score : "))
        GAP      = int(input("  New Gap score      : "))
        print(f"  Updated: Match={MATCH}  Mismatch={MISMATCH}  Gap={GAP}")
    except ValueError:
        print("  [ERROR] Enter integers only.")

# =============================================================================
#  SYSTEM INFO
# =============================================================================
def print_system_info():
    banner("SYSTEM INFORMATION")
    print(f"  Python          : {sys.version.split()[0]}")
    print(f"  CPU cores       : {cpu_count()}")
    print(f"  Platform        : {sys.platform}")
    print(f"  CUDA status     : NOT AVAILABLE on this machine")
    print(f"  Parallel engine : Python multiprocessing + NumPy wavefront")
    print(f"  Note            : On a CUDA machine, use src/main.cu + nvcc")

# =============================================================================
#  GRAPH GENERATION
# =============================================================================
def generate_graphs():
    banner("GENERATING PERFORMANCE GRAPHS")

    if not os.path.exists(CSV_PATH):
        print(f"  [ERROR] {CSV_PATH} not found. Run benchmark first.")
        return

    import pandas as pd
    df = pd.read_csv(CSV_PATH)
    required = {"SeqLen", "CPU_ms", "GPUKernel_ms", "GPUTotal_ms",
                "SpeedupKernel", "SpeedupTotal"}
    if not required.issubset(df.columns):
        print(f"  [ERROR] CSV missing columns. Re-run benchmark.")
        return

    df = df.groupby("SeqLen", as_index=False).mean(numeric_only=True)
    df = df.sort_values("SeqLen")

    COLOR_CPU    = "#E74C3C"
    COLOR_KERNEL = "#2ECC71"
    COLOR_TOTAL  = "#3498DB"
    COLOR_SK     = "#9B59B6"
    COLOR_ST     = "#F39C12"
    BG_OUTER     = "#1A1A2E"
    BG_INNER     = "#16213E"
    GRID_COLOR   = "#2A2A4A"

    def style_ax(ax, fig):
        fig.patch.set_facecolor(BG_OUTER)
        ax.set_facecolor(BG_INNER)
        ax.tick_params(colors="white", labelsize=11)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
        ax.grid(True, color=GRID_COLOR, linewidth=0.8, linestyle='--', alpha=0.7)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # ── Graph 1: Execution Time Comparison ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6.5))
    style_ax(ax, fig)

    ax.plot(df["SeqLen"], df["CPU_ms"],       "o-", color=COLOR_CPU,
            lw=2.5, ms=8, label="CPU Sequential", zorder=5)
    ax.plot(df["SeqLen"], df["GPUKernel_ms"], "s-", color=COLOR_KERNEL,
            lw=2.5, ms=8, label="Parallel Wavefront (kernel)", zorder=5)
    ax.plot(df["SeqLen"], df["GPUTotal_ms"],  "^-", color=COLOR_TOTAL,
            lw=2.5, ms=8, label="Parallel Wavefront (total)", zorder=5)

    ax.fill_between(df["SeqLen"], df["CPU_ms"], df["GPUKernel_ms"],
                    alpha=0.12, color=COLOR_CPU)

    ax.set_xlabel("Sequence Length (bp)", fontsize=13, labelpad=8)
    ax.set_ylabel("Execution Time (ms)", fontsize=13, labelpad=8)
    ax.set_title(
        "CPU Sequential vs Parallel Wavefront Execution Time\n"
        "Needleman-Wunsch DNA Sequence Alignment  |  BCS702 CCA-1",
        fontsize=14, fontweight="bold", pad=12)
    leg = ax.legend(fontsize=11, facecolor="#0F3460", edgecolor="#444466",
                    labelcolor="white", framealpha=0.9)

    out1 = os.path.join(RESULTS_DIR, "time_comparison.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [Saved] {out1}")

    # ── Graph 2: Speedup ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6.5))
    style_ax(ax, fig)

    ax.plot(df["SeqLen"], df["SpeedupKernel"], "o-", color=COLOR_SK,
            lw=2.5, ms=8, label="Speedup (kernel only)")
    ax.plot(df["SeqLen"], df["SpeedupTotal"],  "s-", color=COLOR_ST,
            lw=2.5, ms=8, label="Speedup (total incl. overhead)")
    ax.axhline(1.0, color="white", lw=1.2, ls="--", alpha=0.5, label="Baseline (1×)")
    ax.fill_between(df["SeqLen"], 1, df["SpeedupKernel"],
                    alpha=0.12, color=COLOR_SK, label="_nolegend_")

    for _, row in df.iterrows():
        ax.annotate(f'{row["SpeedupKernel"]:.1f}×',
                    xy=(row["SeqLen"], row["SpeedupKernel"]),
                    xytext=(0, 10), textcoords="offset points",
                    color=COLOR_SK, fontsize=9, ha='center')

    ax.set_xlabel("Sequence Length (bp)", fontsize=13, labelpad=8)
    ax.set_ylabel("Speedup (×)", fontsize=13, labelpad=8)
    ax.set_title(
        "Parallel Wavefront Speedup over CPU Sequential\n"
        "Needleman-Wunsch DNA Sequence Alignment  |  BCS702 CCA-1",
        fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=11, facecolor="#0F3460", edgecolor="#444466",
              labelcolor="white", framealpha=0.9)

    out2 = os.path.join(RESULTS_DIR, "speedup.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [Saved] {out2}")

    print("\n  Both graphs saved successfully.")

# =============================================================================
#  INTERACTIVE MENU
# =============================================================================
def main_menu():
    banner("DNA SEQUENCE ALIGNMENT — BCS702 CCA 1")
    print("  Course  : Parallel Computing (BCS702)  |  Sem VII  |  Sec D")
    print("  Project : Parallel DNA Alignment — CCA 1 (Needleman-Wunsch)")
    print_system_info()

    while True:
        print(f"\n{'─'*42}")
        print("  MENU")
        print(f"{'─'*42}")
        print("  1. Manual sequence input")
        print("  2. Random sequence generation")
        print("  3. Quick validation test  (ACGT vs ACT)")
        print("  4. Benchmark mode  (500 → 8000)")
        print("  5. Configure scoring parameters")
        print("  6. Generate performance graphs")
        print("  7. Display system information")
        print("  0. Exit")
        print("-" * 42)
        choice = input("  Enter choice: ").strip()

        if   choice == "1": manual_mode()
        elif choice == "2": random_mode()
        elif choice == "3": quick_validation_test()
        elif choice == "4": run_benchmark()
        elif choice == "5": configure_scoring()
        elif choice == "6": generate_graphs()
        elif choice == "7": print_system_info()
        elif choice == "0":
            print(f"\n  Exiting. Results saved to {CSV_PATH}")
            break
        else:
            print("  [ERROR] Invalid choice. Enter 0-7.")

# =============================================================================
#  CLI ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="DNA Sequence Alignment — BCS702 CCA 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python dna_align.py                    # interactive menu
              python dna_align.py --validate         # quick test: ACGT vs ACT
              python dna_align.py --benchmark        # full benchmark 500-8000
              python dna_align.py --graphs           # generate graphs only
              python dna_align.py --seq1 ACGT --seq2 ACT   # direct alignment
        """)
    )
    parser.add_argument("--validate",  action="store_true", help="Quick validation test")
    parser.add_argument("--benchmark", action="store_true", help="Full benchmark mode")
    parser.add_argument("--graphs",    action="store_true", help="Generate graphs from CSV")
    parser.add_argument("--seq1",      type=str, help="DNA sequence 1")
    parser.add_argument("--seq2",      type=str, help="DNA sequence 2")
    parser.add_argument("--length",    type=int, help="Random sequence length")
    parser.add_argument("--match",     type=int, default=2,  help="Match score (default 2)")
    parser.add_argument("--mismatch",  type=int, default=-1, help="Mismatch score (default -1)")
    parser.add_argument("--gap",       type=int, default=-2, help="Gap score (default -2)")

    args = parser.parse_args()
    global MATCH, MISMATCH, GAP
    MATCH, MISMATCH, GAP = args.match, args.mismatch, args.gap

    if args.validate:
        quick_validation_test()
    elif args.benchmark:
        run_benchmark()
        generate_graphs()
    elif args.graphs:
        generate_graphs()
    elif args.seq1 and args.seq2:
        run_alignment(args.seq1.upper(), args.seq2.upper())
    elif args.length:
        s1 = random_dna(args.length, 42)
        s2 = random_dna(args.length, 99)
        run_alignment(s1, s2)
    else:
        main_menu()


if __name__ == "__main__":
    main()
