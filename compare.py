#!/usr/bin/env python3
# =============================================================================
#  compare.py  —  DNA Alignment: Python/CPU  vs  CUDA/GPU
# =============================================================================
#  Course  : Parallel Computing (BCS702) | Semester VII | Section D
#  Project : Parallel DNA Sequence Alignment Using CUDA — CCA 1
#
#  ARCHITECTURE:
#
#               DNA ALIGNMENT
#                    │
#        ┌───────────┴───────────┐
#        ↓                       ↓
#  Python/CPU              CUDA/C++ (src/main.cu)
#        │                       │
#        ↓                       ↓
# Needleman-Wunsch        Needleman-Wunsch
#   (Sequential)            (Wavefront Kernel)
#        │                       │
#        └───────────┬───────────┘
#                    ↓
#             Compare Results
#                    ↓
#          CPU vs CUDA Speedup
#
#  USAGE:
#    python compare.py                      # full comparison menu
#    python compare.py --validate           # quick validation test
#    python compare.py --benchmark          # run both sides + compare
#    python compare.py --load-cuda FILE.csv # load real Colab CUDA results
# =============================================================================

import os, sys, io, csv, time, random, argparse, textwrap
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(ROOT, 'results')
CPU_CSV      = os.path.join(RESULTS_DIR, 'cpu_benchmark.csv')
CUDA_CSV     = os.path.join(RESULTS_DIR, 'cuda_benchmark.csv')   # from Colab
CMP_CSV      = os.path.join(RESULTS_DIR, 'comparison.csv')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Scoring ────────────────────────────────────────────────────────────────────
MATCH    =  2
MISMATCH = -1
GAP      = -2

# ══════════════════════════════════════════════════════════════════════════════
#  BRANCH A:  Python / CPU
#  Sequential Needleman-Wunsch — identical recurrence to CUDA C++ version
# ══════════════════════════════════════════════════════════════════════════════
def cpu_nw(seq1: str, seq2: str) -> int:
    """
    Sequential Needleman-Wunsch.
    Fills (m+1)×(n+1) DP matrix row-by-row, left-to-right.
    Recurrence:
      dp[i][j] = max(
        dp[i-1][j-1] + score(seq1[i], seq2[j]),   # diagonal
        dp[i-1][j]   + GAP,                         # up
        dp[i][j-1]   + GAP                          # left
      )
    """
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m+1, n+1), dtype=np.int32)
    dp[0, :] = np.arange(n+1) * GAP
    dp[:, 0] = np.arange(m+1) * GAP

    for i in range(1, m+1):
        for j in range(1, n+1):
            d = dp[i-1, j-1] + (MATCH if seq1[i-1] == seq2[j-1] else MISMATCH)
            u = dp[i-1, j  ] + GAP
            l = dp[i,   j-1] + GAP
            dp[i, j] = max(d, u, l)

    return int(dp[m, n])


# ══════════════════════════════════════════════════════════════════════════════
#  BRANCH B:  CUDA / C++  (simulation when GPU not available)
#  Uses the same anti-diagonal wavefront ordering as the CUDA kernel.
#  On a real GPU machine: compile src/main.cu with nvcc and use those numbers.
# ══════════════════════════════════════════════════════════════════════════════
def cuda_nw_sim(seq1: str, seq2: str) -> int:
    """
    Python simulation of the CUDA wavefront kernel.
    Same anti-diagonal sweep as nwKernel in src/main.cu:
      for diag in 2 .. m+n:           # ← one kernel launch per diagonal
          compute all (i,j | i+j=diag) simultaneously  ← CUDA threads
          barrier                      ← cudaDeviceSynchronize()
    Uses fully-vectorised numpy — no Python inner loop.
    """
    m, n = len(seq1), len(seq2)
    arr1 = np.frombuffer(seq1.encode(), dtype=np.uint8)
    arr2 = np.frombuffer(seq2.encode(), dtype=np.uint8)
    dp = np.zeros((m+1, n+1), dtype=np.int32)
    dp[0, :] = np.arange(n+1) * GAP
    dp[:, 0] = np.arange(m+1) * GAP

    for diag in range(2, m + n + 1):              # ← kernel launch loop
        rows = np.arange(max(1, diag-n), min(m, diag-1)+1, dtype=np.intp)
        cols = diag - rows
        mm   = (arr1[rows-1] == arr2[cols-1])
        sd   = dp[rows-1, cols-1] + np.where(mm, MATCH, MISMATCH)
        su   = dp[rows-1, cols  ] + GAP
        sl   = dp[rows,   cols-1] + GAP
        dp[rows, cols] = np.maximum(sd, np.maximum(su, sl))
        # ↑ numpy vectorisation = all threads complete; barrier implicit

    return int(dp[m, n])


# ── Helpers ────────────────────────────────────────────────────────────────────
def random_dna(n: int, seed: int = 42) -> str:
    rng = random.Random(seed)
    return ''.join(rng.choice('ACGT') for _ in range(n))

def timed(fn, *args):
    """Returns (result, elapsed_ms)."""
    t0 = time.perf_counter()
    r  = fn(*args)
    return r, (time.perf_counter() - t0) * 1000.0

def banner(title: str):
    line = '═' * 60
    print(f'\n{line}\n  {title}\n{line}')

def section(title: str):
    print(f'\n  {"─"*54}')
    print(f'  {title}')
    print(f'  {"─"*54}')


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION — confirm both branches give identical scores
# ══════════════════════════════════════════════════════════════════════════════
def validate():
    banner('VALIDATION — Python/CPU  vs  CUDA/C++ (simulated)')

    tests = [
        ('ACGT',     'ACT'),
        ('ACGTACGT', 'ACGTTCGT'),
        ('GATTACA',  'GCATGCU'.replace('U','T')),
        ('AAAA',     'TTTT'),
    ]

    all_pass = True
    for s1, s2 in tests:
        cpu_score,  _ = timed(cpu_nw,       s1, s2)
        cuda_score, _ = timed(cuda_nw_sim,  s1, s2)
        ok = (cpu_score == cuda_score)
        if not ok: all_pass = False
        status = 'PASS ✓' if ok else 'FAIL ✗'
        print(f'  {s1:12s}  vs  {s2:12s}  →  CPU={cpu_score:4d}  CUDA={cuda_score:4d}  [{status}]')

    # Show full DP matrix for first test
    print('\n  Full DP Matrix (ACGT vs ACT):')
    s1, s2 = 'ACGT', 'ACT'
    m, n = len(s1), len(s2)
    dp = np.zeros((m+1, n+1), dtype=np.int32)
    dp[0,:] = np.arange(n+1) * GAP
    dp[:,0] = np.arange(m+1) * GAP
    for i in range(1,m+1):
        for j in range(1,n+1):
            d=dp[i-1,j-1]+(MATCH if s1[i-1]==s2[j-1] else MISMATCH)
            dp[i,j]=max(d, dp[i-1,j]+GAP, dp[i,j-1]+GAP)
    hdr = '       ' + ''.join(f'  {c:>3}' for c in ['-']+list(s2))
    print(hdr)
    for i, lbl in enumerate(['-']+list(s1)):
        print(f'  {lbl}   ' + ''.join(f'  {dp[i,j]:>3}' for j in range(n+1)))

    print(f'\n  Overall Validation : {"ALL PASS ✓" if all_pass else "SOME FAIL ✗"}')
    return all_pass


# ══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK — run both branches, compare, save CSV
# ══════════════════════════════════════════════════════════════════════════════
def run_benchmark(sizes=None, trials=3):
    if sizes is None:
        sizes = [500, 1000, 2000, 4000, 8000]

    banner('BENCHMARK — Python/CPU  vs  CUDA/C++ (Wavefront Sim)')
    print('  NOTE: CUDA column uses Python wavefront simulation (same algorithm).')
    print('        For REAL GPU times → run DNA_CUDA_BCS702_CCA1.ipynb on Google Colab.')
    print('        Then load with: python compare.py --load-cuda results.csv\n')

    # Write CPU CSV header
    with open(CPU_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['SeqLen','CPU_ms','Score'])

    rows = []
    print(f"  {'Size':>6}  {'CPU (ms)':>12}  {'CUDA-Sim (ms)':>14}  {'Speedup':>9}  {'Valid':>6}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*14}  {'─'*9}  {'─'*6}")

    for sz in sizes:
        cpu_total = cuda_total = 0.0
        last_cpu = last_cuda = 0
        all_pass = True

        for t in range(trials):
            s1 = random_dna(sz, 100 + t)
            s2 = random_dna(sz, 200 + t)

            # ── Branch A: Python/CPU ──────────────────────────
            last_cpu,  cpu_ms  = timed(cpu_nw,      s1, s2)

            # ── Branch B: CUDA wavefront sim ──────────────────
            last_cuda, cuda_ms = timed(cuda_nw_sim, s1, s2)

            cpu_total  += cpu_ms
            cuda_total += cuda_ms
            if last_cpu != last_cuda:
                all_pass = False

        avg_cpu  = cpu_total  / trials
        avg_cuda = cuda_total / trials
        speedup  = avg_cpu / avg_cuda if avg_cuda > 0 else 0
        status   = 'PASS' if all_pass else 'FAIL'

        print(f"  {sz:>6}  {avg_cpu:>12.2f}  {avg_cuda:>14.2f}  {speedup:>8.3f}x  {status:>6}")

        with open(CPU_CSV, 'a', newline='') as f:
            csv.writer(f).writerow([sz, f'{avg_cpu:.4f}', last_cpu])

        rows.append({
            'SeqLen':       sz,
            'CPU_ms':       round(avg_cpu,  3),
            'CUDA_ms':      round(avg_cuda, 3),
            'SpeedupSim':   round(speedup,  3),
            'CPU_Score':    last_cpu,
            'CUDA_Score':   last_cuda,
            'Validation':   status,
            'CUDA_Source':  'Simulation'
        })

    _save_comparison(rows)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD REAL CUDA RESULTS (from Google Colab) and merge with CPU
# ══════════════════════════════════════════════════════════════════════════════
def load_cuda_results(cuda_csv_path: str, trials=3):
    banner('MERGE — Python/CPU  +  Real CUDA GPU Results')
    print(f'  Loading CUDA results from: {cuda_csv_path}')

    import csv as _csv
    cuda_rows = {}
    with open(cuda_csv_path, newline='') as f:
        for row in _csv.DictReader(f):
            sz = int(row['SeqLen'])
            cuda_rows[sz] = {
                'kernel_ms': float(row['GPUKernel_ms']),
                'total_ms':  float(row['GPUTotal_ms']),
                'speedup_k': float(row['SpeedupKernel']),
                'speedup_t': float(row['SpeedupTotal']),
                'score':     int(row['GPU_Score']),
                'valid':     row['Validation']
            }

    sizes = sorted(cuda_rows.keys())
    rows  = []

    print(f"\n  {'Size':>6}  {'CPU (ms)':>12}  {'GPU Kernel':>12}  {'GPU Total':>10}  {'Speedup-K':>10}  {'Valid':>6}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*6}")

    for sz in sizes:
        # Re-run CPU for this size
        cpu_total = 0.0
        last_cpu  = 0
        for t in range(trials):
            s1 = random_dna(sz, 100+t)
            s2 = random_dna(sz, 200+t)
            last_cpu, cpu_ms = timed(cpu_nw, s1, s2)
            cpu_total += cpu_ms
        avg_cpu = cpu_total / trials

        c = cuda_rows[sz]
        print(f"  {sz:>6}  {avg_cpu:>12.2f}  {c['kernel_ms']:>12.2f}  {c['total_ms']:>10.2f}  {c['speedup_k']:>9.2f}x  {c['valid']:>6}")

        rows.append({
            'SeqLen':       sz,
            'CPU_ms':       round(avg_cpu, 3),
            'CUDA_ms':      round(c['kernel_ms'], 3),
            'CUDATotal_ms': round(c['total_ms'], 3),
            'SpeedupKernel':c['speedup_k'],
            'SpeedupTotal': c['speedup_t'],
            'CPU_Score':    last_cpu,
            'CUDA_Score':   c['score'],
            'Validation':   c['valid'],
            'CUDA_Source':  'Real GPU (Colab T4)'
        })

    # Save merged CSV
    with open(CUDA_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f'\n  ✓ Merged results saved to {CUDA_CSV}')

    _save_comparison(rows)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE COMPARISON CSV
# ══════════════════════════════════════════════════════════════════════════════
def _save_comparison(rows: list):
    with open(CMP_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f'\n  [CSV] Comparison results saved → {CMP_CSV}')


# ══════════════════════════════════════════════════════════════════════════════
#  GENERATE COMPARISON GRAPHS
# ══════════════════════════════════════════════════════════════════════════════
def generate_graphs(rows: list, cuda_source: str = 'Simulation'):
    banner('GENERATING PERFORMANCE GRAPHS')

    sizes     = [r['SeqLen']  for r in rows]
    cpu_times = [r['CPU_ms']  for r in rows]
    cuda_times= [r['CUDA_ms'] for r in rows]

    # Determine speedup column
    if 'SpeedupKernel' in rows[0]:
        speedups = [r['SpeedupKernel'] for r in rows]
    else:
        speedups = [r.get('SpeedupSim', 0) for r in rows]

    # ── Styling ────────────────────────────────────────────────
    BG     = '#0d1117'
    BG2    = '#161b22'
    GRID   = '#21262d'
    CPU_C  = '#fc8181'
    CUDA_C = '#48bb78'
    SP_C   = '#9f7aea'
    GOLD   = '#f6e05e'

    def stylize(ax, fig):
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG2)
        ax.tick_params(colors='#8b949e', labelsize=11)
        ax.xaxis.label.set_color('#c9d1d9')
        ax.yaxis.label.set_color('#c9d1d9')
        ax.title.set_color('#f0f6fc')
        for s in ax.spines.values(): s.set_edgecolor('#30363d')
        ax.grid(True, color=GRID, lw=0.8, ls='--', alpha=0.8)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))

    # ── Graph 1: Execution Time ───────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    stylize(ax, fig)

    ax.plot(sizes, cpu_times,  'o-', color=CPU_C,  lw=2.5, ms=9, label='Python / CPU (Sequential)', zorder=5)
    ax.plot(sizes, cuda_times, 's-', color=CUDA_C, lw=2.5, ms=9, label=f'CUDA / C++ ({cuda_source})',    zorder=5)
    ax.fill_between(sizes, cpu_times, cuda_times, alpha=0.12, color=CPU_C)

    for x, y in zip(sizes, cpu_times):
        ax.annotate(f'{y:.0f}', (x,y), xytext=(0,12), textcoords='offset points',
                    color=CPU_C, fontsize=9, ha='center')
    for x, y in zip(sizes, cuda_times):
        ax.annotate(f'{y:.1f}', (x,y), xytext=(0,-18), textcoords='offset points',
                    color=CUDA_C, fontsize=9, ha='center')

    ax.set_xlabel('Sequence Length (bp)', fontsize=13, labelpad=8)
    ax.set_ylabel('Execution Time (ms)',  fontsize=13, labelpad=8)
    ax.set_title(
        'DNA Alignment: Python/CPU  vs  CUDA/C++ Execution Time\n'
        f'Needleman-Wunsch  |  BCS702 CCA-1  |  CUDA Source: {cuda_source}',
        fontsize=14, fontweight='bold', pad=14)
    ax.legend(fontsize=12, facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9')
    plt.tight_layout()
    out1 = os.path.join(RESULTS_DIR, 'comparison_time.png')
    plt.savefig(out1, dpi=150, facecolor=BG)
    plt.close()
    print(f'  [✓] Saved: {out1}')

    # ── Graph 2: Speedup ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    stylize(ax, fig)

    bar_colors = [CUDA_C if s >= 1 else CPU_C for s in speedups]
    bars = ax.bar(sizes, speedups, color=bar_colors, alpha=0.75,
                  width=[sz*0.08 for sz in sizes], edgecolor='#30363d', linewidth=0.8)
    ax.plot(sizes, speedups, 'o-', color=GOLD, lw=2, ms=8, zorder=5, label='Speedup trend')
    ax.axhline(1.0, color='#8b949e', lw=1.5, ls='--', alpha=0.7, label='Baseline (1×, no speedup)')
    ax.fill_between(sizes, 1, speedups, alpha=0.08, color=GOLD)

    for x, s in zip(sizes, speedups):
        ax.annotate(f'{s:.2f}×', (x, s),
                    xytext=(0, 10), textcoords='offset points',
                    color=GOLD, fontsize=11, ha='center', fontweight='bold')

    ax.set_xlabel('Sequence Length (bp)', fontsize=13, labelpad=8)
    ax.set_ylabel('Speedup (×)',          fontsize=13, labelpad=8)
    ax.set_title(
        'CUDA GPU Speedup over Python CPU\n'
        f'Needleman-Wunsch  |  BCS702 CCA-1  |  CUDA Source: {cuda_source}',
        fontsize=14, fontweight='bold', pad=14)
    green_patch = mpatches.Patch(color=CUDA_C, alpha=0.75, label='Speedup ≥ 1× (GPU wins)')
    red_patch   = mpatches.Patch(color=CPU_C,  alpha=0.75, label='Speedup < 1× (CPU wins)')
    ax.legend(handles=[green_patch, red_patch,
              plt.Line2D([],[],color=GOLD,lw=2,marker='o',label='Speedup trend')],
              fontsize=11, facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9')
    plt.tight_layout()
    out2 = os.path.join(RESULTS_DIR, 'comparison_speedup.png')
    plt.savefig(out2, dpi=150, facecolor=BG)
    plt.close()
    print(f'  [✓] Saved: {out2}')

    print('\n  Both comparison graphs saved successfully.')


# ══════════════════════════════════════════════════════════════════════════════
#  PRINT FINAL SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(rows: list):
    banner('FINAL COMPARISON SUMMARY')
    src = rows[0].get('CUDA_Source', 'Simulation')

    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  DNA ALIGNMENT — CPU vs CUDA COMPARISON                     │
  │  Algorithm  : Needleman-Wunsch Global Sequence Alignment    │
  │  CPU Branch : Python (Sequential nested loop DP)            │
  │  GPU Branch : CUDA C++ Wavefront Kernel   [{src:<18}]│
  │  Scoring    : Match=+{MATCH}  Mismatch={MISMATCH}  Gap={GAP}                    │
  └─────────────────────────────────────────────────────────────┘
    """)

    hdr = f"  {'Size':>6}  {'CPU (ms)':>10}  {'CUDA (ms)':>10}  {'Speedup':>9}  {'Validate':>8}"
    print(hdr)
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*9}  {'─'*8}")

    for r in rows:
        sp_key = 'SpeedupKernel' if 'SpeedupKernel' in r else 'SpeedupSim'
        sp = r.get(sp_key, 0)
        arrow = '⬆' if sp > 1 else '⬇'
        print(f"  {r['SeqLen']:>6}  {r['CPU_ms']:>10.2f}  {r['CUDA_ms']:>10.2f}  {sp:>8.2f}x{arrow}  {r['Validation']:>8}")

    if rows:
        sp_key = 'SpeedupKernel' if 'SpeedupKernel' in rows[0] else 'SpeedupSim'
        max_sp = max(r.get(sp_key,0) for r in rows)
        max_sz = next(r['SeqLen'] for r in rows if r.get(sp_key,0) == max_sp)
        all_p  = all(r['Validation'] == 'PASS' for r in rows)
        print(f"""
  Peak Speedup   : {max_sp:.2f}× at {max_sz} bp
  All Validation : {'PASS ✓' if all_p else 'FAIL ✗'}
  CUDA Source    : {rows[0].get('CUDA_Source','—')}
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE MENU
# ══════════════════════════════════════════════════════════════════════════════
def menu():
    banner('DNA ALIGNMENT — CPU vs CUDA Comparison Tool')
    print('  Course  : Parallel Computing (BCS702) | Sem VII | Sec D')
    print('  Project : CCA 1 — Needleman-Wunsch + CUDA Wavefront')
    print("""
  ARCHITECTURE:
              DNA ALIGNMENT
                   │
       ┌───────────┴───────────┐
       ↓                       ↓
 Python/CPU              CUDA/C++
       │                       │
       ↓                       ↓
Needleman-Wunsch        Needleman-Wunsch
  (Sequential)            (Wavefront Kernel)
       │                       │
       └───────────┬───────────┘
                   ↓
            Compare Results
                   ↓
         CPU vs CUDA Speedup
    """)

    while True:
        print('  ──────────────────────────────────────────────')
        print('  MENU')
        print('  ──────────────────────────────────────────────')
        print('  1. Validate  — confirm CPU == CUDA (4 test cases)')
        print('  2. Benchmark — run CPU + CUDA sim, compare (500–8000 bp)')
        print('  3. Load real CUDA results from Google Colab CSV')
        print('  4. Generate comparison graphs')
        print('  0. Exit')
        print('  ──────────────────────────────────────────────')
        choice = input('  Enter choice: ').strip()

        if choice == '1':
            validate()
        elif choice == '2':
            rows = run_benchmark()
            print_summary(rows)
            generate_graphs(rows, 'Simulation')
        elif choice == '3':
            p = input('  Path to CUDA CSV (from Colab): ').strip()
            if not os.path.exists(p):
                print(f'  [ERROR] File not found: {p}')
            else:
                rows = load_cuda_results(p)
                print_summary(rows)
                generate_graphs(rows, 'Real GPU (Colab T4)')
        elif choice == '4':
            if os.path.exists(CMP_CSV):
                import csv as _csv
                with open(CMP_CSV, newline='') as f:
                    rows = list(_csv.DictReader(f))
                for r in rows:
                    for k in r:
                        try: r[k] = float(r[k])
                        except: pass
                generate_graphs(rows, rows[0].get('CUDA_Source','—'))
            else:
                print('  [ERROR] No comparison data. Run option 2 or 3 first.')
        elif choice == '0':
            print(f'\n  Exiting. Results saved to {RESULTS_DIR}')
            break
        else:
            print('  [ERROR] Invalid choice.')


# ══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='DNA Alignment — Python/CPU vs CUDA/C++ Comparison | BCS702 CCA-1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python compare.py                             # interactive menu
              python compare.py --validate                  # 4 test cases
              python compare.py --benchmark                 # full benchmark
              python compare.py --load-cuda results.csv    # merge real GPU data
        """)
    )
    parser.add_argument('--validate',   action='store_true', help='Quick validation test')
    parser.add_argument('--benchmark',  action='store_true', help='Full benchmark (500-8000 bp)')
    parser.add_argument('--load-cuda',  type=str, metavar='CSV', help='Load real CUDA results from Colab')
    parser.add_argument('--graphs',     action='store_true', help='Generate graphs from saved CSV')

    args = parser.parse_args()

    if args.validate:
        validate()
    elif args.benchmark:
        rows = run_benchmark()
        print_summary(rows)
        generate_graphs(rows, 'Simulation')
    elif args.load_cuda:
        rows = load_cuda_results(args.load_cuda)
        print_summary(rows)
        generate_graphs(rows, 'Real GPU (Colab T4)')
    elif args.graphs:
        import csv as _csv
        if not os.path.exists(CMP_CSV):
            print(f'[ERROR] {CMP_CSV} not found. Run --benchmark or --load-cuda first.')
            sys.exit(1)
        with open(CMP_CSV, newline='') as f:
            rows = list(_csv.DictReader(f))
        generate_graphs(rows, rows[0].get('CUDA_Source','—'))
    else:
        menu()


if __name__ == '__main__':
    main()
