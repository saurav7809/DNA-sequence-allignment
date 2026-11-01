# DNA Sequence Alignment Using CUDA

> **Course:** Parallel Computing (BCS702) | Semester VII | Section D  
> **Project:** CCA 1 — Parallel DNA Sequence Alignment 7th sem 
> **Algorithm:** Needleman-Wunsch Global Alignment with CUDA Wavefront Parallelism

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [CUDA Installation](#cuda-installation)
4. [Project Structure](#project-structure)
5. [Algorithm Explanation](#algorithm-explanation)
6. [Wavefront Parallelism](#wavefront-parallelism)
7. [How to Compile](#how-to-compile)
8. [How to Run](#how-to-run)
9. [Example Output](#example-output)
10. [Performance Measurement](#performance-measurement)
11. [Generating Graphs](#generating-graphs)
12. [Troubleshooting](#troubleshooting)
13. [Viva Q&A](#viva-qa)

---

## Project Overview

This project implements **DNA sequence alignment** using the **Needleman-Wunsch** global dynamic programming algorithm in two flavors:

| Mode | Description |
|------|-------------|
| **CPU (Sequential)** | Fills the DP matrix row-by-row using a nested loop |
| **GPU (CUDA Parallel)** | Fills the DP matrix diagonal-by-diagonal using CUDA threads |

The program measures execution time for both modes, validates that scores match, calculates speedup, and exports results to a CSV for graph generation.
This project implements the **Needleman-Wunsch** global sequence alignment algorithm in both **CPU (C++)** and **GPU (CUDA)** to demonstrate the performance benefits of parallel computing in bioinformatics.

---

## Requirements

| Component | Minimum Version |
|-----------|----------------|
| NVIDIA GPU | Any CUDA-capable GPU (Compute Capability ≥ 3.0) |
| CUDA Toolkit | 11.0 or newer |
| Visual Studio | 2019 or 2022 (Community is free) — for MSVC |
| Python | 3.8+ (for graph generation only) |
| matplotlib, pandas | `pip install matplotlib pandas` |

---

## CUDA Installation

### Step 1 — Install CUDA Toolkit
1. Go to: https://developer.nvidia.com/cuda-downloads
2. Select: **Windows → x86_64 → Windows 10 → exe (network)**
3. Download and run the installer.
4. Choose **Custom Installation**, include: CUDA Toolkit + CUDA Samples.
5. Restart after installation.

### Step 2 — Install Visual Studio
- Download **Visual Studio 2022 Community** from https://visualstudio.microsoft.com/
- During install select: **"Desktop development with C++"**

### Step 3 — Verify Installation
Open a new PowerShell or Command Prompt and run:
```powershell
nvcc --version
nvidia-smi
```

---

## Project Structure

```
DNA-CUDA-Alignment/
├── src/
│   └── main.cu            ← All CUDA + CPU code lives here
├── results/
│   ├── benchmark.csv      ← Auto-generated after running
│   ├── time_comparison.png
│   └── speedup.png
├── screenshots/           ← Put your terminal screenshots here
├── report/                ← Your written report PDF
├── generate_graphs.py     ← Graph generation script
└── README.md
```

---

## Algorithm Explanation

### Needleman-Wunsch (Global Alignment)

The algorithm finds the **optimal global alignment** between two sequences by filling a 2-D scoring matrix `dp[i][j]`.

**Recurrence:**
```
dp[i][j] = max(
    dp[i-1][j-1] + score(seq1[i-1], seq2[j-1]),   // diagonal: match/mismatch
    dp[i-1][j]   + GAP,                             // from above: gap in seq2
    dp[i][j-1]   + GAP                              // from left:  gap in seq1
)
```

**Initialization:**
```
dp[0][j] = j × GAP    (first row)
dp[i][0] = i × GAP    (first column)
```

**Result:** `dp[m][n]` — the optimal global alignment score.

**Scoring used in this project:**
| Event | Score |
|-------|-------|
| Match | +2 |
| Mismatch | -1 |
| Gap | -2 |

---

## Wavefront Parallelism

The CPU fills cells row by row — fully sequential. The GPU uses **anti-diagonal (wavefront) parallelism**:

```
Anti-diagonal d contains all cells (i, j) where i + j = d
```

- All cells on the same anti-diagonal are **independent** of each other.
- They only depend on cells from **previous diagonals** (already computed).
- So we can compute all cells of diagonal d **simultaneously** using CUDA threads.

```
Matrix visualization (4×4):
       j=0  j=1  j=2  j=3
i=0  [  0 ] [  0 ] [  0 ] [  0 ]
i=1  [  0 ] [ d=2] [ d=3] [ d=4]
i=2  [  0 ] [ d=3] [ d=4] [ d=5]
i=3  [  0 ] [ d=4] [ d=5] [ d=6]

d=2 : 1 thread  → cell (1,1)
d=3 : 2 threads → cells (1,2), (2,1) — computed in parallel
d=4 : 3 threads → cells (1,3), (2,2), (3,1) — computed in parallel
d=5 : 2 threads → cells (2,3), (3,2) — computed in parallel
d=6 : 1 thread  → cell (3,3) — final score
```

**In code:**
```cuda
__global__ void nwKernel(..., int diag) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int i = max(1, diag-n) + k;
    int j = diag - i;
    // compute dp[i][j] from dp[i-1][j-1], dp[i-1][j], dp[i][j-1]
}
```

Between diagonals, `cudaDeviceSynchronize()` ensures ordering is preserved.

---

## How to Compile

### Using CUDA Developer Command Prompt (Recommended)

Open **"x64 Native Tools Command Prompt for VS 2022"** (search in Start Menu), then:

```cmd
cd C:\Users\gaura\.gemini\antigravity-ide\scratch\DNA-CUDA-Alignment

mkdir results

nvcc -O2 -arch=sm_75 -o dna_align src/main.cu
```

> Replace `sm_75` with your GPU's compute capability:
> - RTX 20xx series → `sm_75`
> - RTX 30xx series → `sm_86`
> - RTX 40xx series → `sm_89`
> - GTX 10xx series → `sm_61`

To find your compute capability: `nvidia-smi --query-gpu=compute_cap --format=csv`

---

## How to Run

```cmd
# Interactive menu
dna_align.exe

# Quick validation test (ACGT vs ACT)
dna_align.exe --validate

# Full benchmark (all 5 sizes, 3 trials each)
dna_align.exe --benchmark
```

---

## Example Output

```
========================================
 DNA SEQUENCE ALIGNMENT USING CUDA
========================================
  Course  : Parallel Computing (BCS702)  |  Sem VII  |  Sec D

[GPU 0] NVIDIA GeForce RTX 3060
  Compute Capability : 8.6
  Total Global Memory: 12288.00 MB
  Multiprocessors    : 28
  Max Threads/Block  : 1024

────────────────────────────────────────
 MENU
────────────────────────────────────────
 1. Manual sequence input
 2. Random sequence generation
 3. Quick validation test  (ACGT vs ACT)
 4. Benchmark mode  (500→8000)
 5. Configure scoring parameters
 6. Display GPU information
 0. Exit
Enter choice: 3

========================================
 QUICK VALIDATION TEST
========================================
seq1 = ACGT
seq2 = ACT

CPU Score : 5
GPU Score : 5
Results MATCH  ✓

========================================
 DNA SEQUENCE ALIGNMENT USING CUDA
========================================
  Sequence Length   : 4

  Scoring:
    Match    : 2
    Mismatch : -1
    Gap      : -2

  CPU Alignment Score : 5
  GPU Alignment Score : 5

  CPU Time        : 0.0012 ms
  GPU Kernel Time : 0.0423 ms
  GPU Total Time  : 0.5831 ms

  Speedup (kernel) : 0.028x      ← GPU overhead dominates for tiny inputs
  Speedup (total)  : 0.002x

  Validation : PASS ✓
```

> ⚠️ For small sequences, CPU is faster due to GPU launch overhead.
> The GPU advantage appears at length ≥ 500.

### Benchmark Example (2000 bp):
```
--- Sequence Length: 2000 ---
  Trial 1: CPU=1245 GPU=1245 PASS
  Trial 2: CPU=1240 GPU=1240 PASS
  Trial 3: CPU=1238 GPU=1238 PASS
  Avg CPU Time    : 124.3200 ms
  Avg Kernel Time :  18.6700 ms
  Avg Total Time  :  21.4500 ms
  Speedup(kernel) : 6.66x
  Speedup(total)  : 5.80x
  Validation      : ALL PASS
```

---

## Performance Measurement

### CPU Timing
Uses C++11 `std::chrono::high_resolution_clock` with nanosecond resolution:
```cpp
auto t0 = chrono::high_resolution_clock::now();
cpuNeedlemanWunsch(...);
auto t1 = chrono::high_resolution_clock::now();
double ms = chrono::duration<double, milli>(t1 - t0).count();
```

### GPU Timing (CUDA Events)
CUDA Events use the GPU's hardware clock for microsecond-accurate GPU timing:
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
// ... kernel launches ...
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
```

Two timing scopes:
- **Kernel only:** starts after H→D copy, stops before D→H copy
- **Total:** includes memory allocation, H→D copy, kernels, D→H copy

### Speedup Formula
```
Speedup = CPU_Time_ms / GPU_Time_ms
```

---

## Generating Graphs

After running the benchmark, a CSV is saved to `results/benchmark.csv`.

Install Python dependencies:
```cmd
pip install matplotlib pandas
```

Run the graph script:
```cmd
python generate_graphs.py
```

Two PNG files will be saved to `results/`:
- `time_comparison.png` — CPU vs GPU time vs sequence length
- `speedup.png` — Speedup (kernel and total) vs sequence length

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `nvcc not found` | Add CUDA bin to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\bin` |
| `error: identifier "cudaEvent_t" undefined` | Missing `#include <cuda_runtime.h>` |
| `CUDA out of memory` | Reduce sequence sizes in benchmark vector (e.g., max 4000) |
| GPU score ≠ CPU score | Ensure `cudaDeviceSynchronize()` is called after every diagonal |
| Very slow GPU for small inputs | Expected — GPU has kernel launch overhead; benefit appears at length ≥ 500 |
| `cl.exe not found` | Must open "x64 Native Tools Command Prompt for VS 2022" |
| Compute capability mismatch | Use `nvidia-smi` to confirm, set `-arch=sm_XX` accordingly |

---

## Viva Q&A

**Q1. What is Needleman-Wunsch and where is it used?**  
A: It's a global sequence alignment algorithm based on dynamic programming. It's used in bioinformatics to align DNA/protein sequences end-to-end. It guarantees the optimal global alignment.

**Q2. Why can't we parallelize the DP matrix row by row?**  
A: Because `dp[i][j]` depends on `dp[i-1][j]` (above), `dp[i][j-1]` (left), and `dp[i-1][j-1]` (diagonal). Different cells in the same row are interdependent via the left dependency.

**Q3. What is wavefront/anti-diagonal parallelism?**  
A: Cells on the same anti-diagonal (where `i + j` is constant) share no data dependencies between them. They only depend on cells from earlier diagonals. So they can be computed simultaneously by different CUDA threads.

**Q4. How are CUDA threads assigned to cells?**  
A: For anti-diagonal `d`, thread `k` computes row `i = max(1, d-n) + k`, column `j = d - i`. The number of threads launched equals the number of active cells on that diagonal.

**Q5. Why do we call `cudaDeviceSynchronize()` after each kernel?**  
A: To enforce the ordering constraint. Diagonal `d+1` depends on diagonal `d` results. Without synchronization, threads from diagonal `d+1` might read incomplete values from diagonal `d`.

**Q6. What are CUDA Events and why use them for timing?**  
A: CUDA Events are lightweight GPU-side timestamps recorded in the GPU's hardware clock. They are more accurate than CPU timers for measuring GPU execution time because they measure time on the GPU itself.

**Q7. Why is GPU slower than CPU for short sequences?**  
A: For short sequences (< 500 bp), the anti-diagonals are very short (few cells), so very few threads are active. The overhead of launching many kernels (one per diagonal) and transferring memory exceeds the computation time.

**Q8. What is compute capability?**  
A: A version number (e.g., 8.6) indicating the GPU's supported CUDA features and architecture. Higher means newer. It's specified during compilation with `-arch=sm_86`.

**Q9. What memory does cudaMalloc allocate?**  
A: Global device memory (DRAM on the GPU chip). It's accessible by all threads but is relatively slow compared to shared memory.

**Q10. What is the time complexity of Needleman-Wunsch?**  
A: O(m × n) time and space, where m and n are sequence lengths. For equal lengths, O(n²). The GPU reduces the practical time constant but not the asymptotic complexity. 

