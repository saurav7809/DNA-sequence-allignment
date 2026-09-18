// =============================================================================
// DNA SEQUENCE ALIGNMENT USING CUDA
// =============================================================================
// Course  : Parallel Computing (BCS702), Semester VII, Section D
// Project : Parallel DNA Sequence Alignment Using CUDA (CCA 1)
// Algorithm: Needleman-Wunsch Global Sequence Alignment
// Author  : [Your Name]
//
// Description:
//   This program implements DNA sequence alignment using the Needleman-Wunsch
//   dynamic programming algorithm.  Two versions are provided:
//     1. Sequential CPU implementation  (baseline)
//     2. Parallel  GPU implementation using NVIDIA CUDA
//
//   The GPU kernel exploits WAVEFRONT / ANTI-DIAGONAL PARALLELISM:
//   cells that lie on the same anti-diagonal of the DP matrix are
//   independent of each other and can be computed concurrently.
//
// Scoring scheme (configurable via menu):
//   Match    = +2
//   Mismatch = -1
//   Gap      =  -2
// =============================================================================

#include <cuda_runtime.h>          // CUDA runtime API (cudaMalloc, cudaMemcpy …)
#include <device_launch_parameters.h>  // blockIdx, threadIdx, etc.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>                  // High-resolution CPU timing
#include <random>                  // Random DNA generation
#include <cstdlib>
#include <cstring>
#include <climits>
#include <cassert>
#include <iomanip>

using namespace std;

// =============================================================================
// SCORING CONSTANTS (modifiable via menu)
// =============================================================================
int MATCH    =  2;
int MISMATCH = -1;
int GAP      = -2;

// =============================================================================
// CUDA ERROR-CHECKING MACRO
// Wraps every CUDA call. On failure it prints file + line and exits.
// =============================================================================
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr,                                                     \
                    "[CUDA ERROR] %s  at %s:%d\n",                             \
                    cudaGetErrorString(err), __FILE__, __LINE__);               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// =============================================================================
// HELPER: print a section header banner
// =============================================================================
static void banner(const string& title) {
    cout << "\n========================================\n";
    cout << " " << title << "\n";
    cout << "========================================\n";
}

// =============================================================================
// HELPER: generate a random DNA string of length n
// Alphabet = {A, C, G, T}
// =============================================================================
static string randomDNA(int n, unsigned seed = 42) {
    static const char bases[] = "ACGT";
    mt19937 rng(seed);
    uniform_int_distribution<int> dist(0, 3);
    string seq(n, ' ');
    for (int i = 0; i < n; ++i) seq[i] = bases[dist(rng)];
    return seq;
}

// =============================================================================
//  GPU INFORMATION DISPLAY
//  Queries and prints device properties: name, memory, compute capability.
// =============================================================================
static void printGPUInfo() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0) {
        cout << "[GPU Info] No CUDA-capable device detected.\n";
        return;
    }

    for (int d = 0; d < deviceCount; ++d) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, d));

        cout << "\n[GPU " << d << "] " << prop.name << "\n";
        cout << "  Compute Capability : " << prop.major << "." << prop.minor << "\n";
        cout << "  Total Global Memory: "
             << (prop.totalGlobalMem / (1024.0 * 1024.0)) << " MB\n";
        cout << "  Multiprocessors    : " << prop.multiProcessorCount << "\n";
        cout << "  Max Threads/Block  : " << prop.maxThreadsPerBlock << "\n";
        cout << "  Warp Size          : " << prop.warpSize << "\n";
    }
}

// =============================================================================
// ─────────────────────────────────────────────────────────────────────────────
//  CPU IMPLEMENTATION  (Sequential Needleman-Wunsch)
// ─────────────────────────────────────────────────────────────────────────────
// =============================================================================

// -----------------------------------------------------------------------------
// cpuNeedlemanWunsch
//   seq1, seq2 : input DNA sequences
//   m          : length of seq1
//   n          : length of seq2
//   returns    : alignment score (bottom-right cell of DP matrix)
// -----------------------------------------------------------------------------
int cpuNeedlemanWunsch(const string& seq1, const string& seq2) {
    int m = (int)seq1.size();
    int n = (int)seq2.size();

    // Allocate a flat 2-D DP table stored in row-major order.
    // dp[i*(n+1)+j] corresponds to DP[i][j].
    vector<int> dp((m + 1) * (n + 1), 0);

    // ---- Initialization ----
    // First row: gaps in seq1 (match seq2 against nothing)
    for (int j = 0; j <= n; ++j) dp[0 * (n + 1) + j] = j * GAP;
    // First column: gaps in seq2
    for (int i = 0; i <= m; ++i) dp[i * (n + 1) + 0] = i * GAP;

    // ---- Fill DP table (sequential, row by row) ----
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            // Diagonal score: match or mismatch
            int scoreDiag = dp[(i - 1) * (n + 1) + (j - 1)]
                            + (seq1[i - 1] == seq2[j - 1] ? MATCH : MISMATCH);
            // Score from top cell: gap in seq2
            int scoreUp   = dp[(i - 1) * (n + 1) + j] + GAP;
            // Score from left cell: gap in seq1
            int scoreLeft = dp[i * (n + 1) + (j - 1)] + GAP;

            dp[i * (n + 1) + j] = max({scoreDiag, scoreUp, scoreLeft});
        }
    }

    return dp[m * (n + 1) + n];  // alignment score
}

// =============================================================================
// ─────────────────────────────────────────────────────────────────────────────
//  CUDA / GPU IMPLEMENTATION  (Parallel Needleman-Wunsch)
// ─────────────────────────────────────────────────────────────────────────────
// =============================================================================

// =============================================================================
// WAVEFRONT / ANTI-DIAGONAL PARALLELISM — THEORY
// =============================================================================
// The Needleman-Wunsch DP cell dp[i][j] depends on:
//   • dp[i-1][j-1]  (diagonal predecessor)
//   • dp[i-1][j]    (cell above)
//   • dp[i][j-1]    (cell to the left)
//
// This means a single row or column CANNOT be processed in parallel.
// However, all cells that share the same anti-diagonal index
//   diag = i + j
// are mutually INDEPENDENT (none depends on another cell in the same diagonal).
//
// Anti-diagonal indices run from 2 (cell [1][1]) to m+n (cell [m][n]).
// For anti-diagonal d, valid (i,j) pairs satisfy:
//   i + j == d,  1 <= i <= m,  1 <= j <= n.
//
// Each CUDA thread in a kernel launch handles ONE cell of the current diagonal.
// We launch one kernel per diagonal and synchronize between them using
// cudaDeviceSynchronize() on the host side.
// =============================================================================

// -----------------------------------------------------------------------------
// CUDA KERNEL: nwKernel
//   Processes one anti-diagonal of the DP matrix.
//
//   Parameters:
//     d_dp   : flat DP matrix on device  (size (m+1)*(n+1))
//     d_seq1 : device copy of seq1
//     d_seq2 : device copy of seq2
//     m, n   : sequence lengths
//     diag   : current anti-diagonal index (= i + j for all cells in this wave)
//     MATCH, MISMATCH, GAP : scoring parameters
// -----------------------------------------------------------------------------
__global__
void nwKernel(int* d_dp, const char* d_seq1, const char* d_seq2,
              int m, int n, int diag,
              int matchScore, int mismatchScore, int gapScore)
{
    // Each thread gets a unique index within the current anti-diagonal.
    // Thread 0 handles the first valid cell, thread 1 the second, and so on.
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    // ---- Determine i and j from anti-diagonal index and thread index ----
    // On anti-diagonal 'diag', i ranges from max(1, diag-n) to min(m, diag-1).
    // Thread k corresponds to: i = start + k,  j = diag - i
    int i_start = max(1, diag - n);   // smallest valid row index on this diag
    int i_end   = min(m, diag - 1);   // largest  valid row index on this diag
    int num_cells = i_end - i_start + 1;  // total cells on this anti-diagonal

    if (k >= num_cells) return;        // guard: excess threads do nothing

    int i = i_start + k;               // row index for this thread
    int j = diag - i;                  // column index (always satisfies i+j=diag)

    // Validate bounds (defensive check)
    if (i < 1 || i > m || j < 1 || j > n) return;

    // DP matrix stride = n+1 (row-major layout)
    int stride = n + 1;

    // Read the three predecessor cells (already computed in earlier diagonals)
    int scoreDiag = d_dp[(i - 1) * stride + (j - 1)]
                    + (d_seq1[i - 1] == d_seq2[j - 1] ? matchScore : mismatchScore);
    int scoreUp   = d_dp[(i - 1) * stride + j] + gapScore;
    int scoreLeft = d_dp[i       * stride + (j - 1)] + gapScore;

    // Write the maximum of the three choices
    d_dp[i * stride + j] = max(scoreDiag, max(scoreUp, scoreLeft));
}

// -----------------------------------------------------------------------------
// cudaNeedlemanWunsch
//   Orchestrates CUDA memory management and kernel launches.
//
//   Returns the alignment score via the output parameter 'score'.
//   kernelMs  : CUDA kernel-only time (milliseconds) via CUDA events
//   totalMs   : total GPU time including memory transfers (milliseconds)
// -----------------------------------------------------------------------------
int cudaNeedlemanWunsch(const string& seq1, const string& seq2,
                        float& kernelMs, float& totalMs)
{
    int m = (int)seq1.size();
    int n = (int)seq2.size();
    int dpSize = (m + 1) * (n + 1);

    // ── Host-side DP buffer (for initialization and final readback) ──────────
    vector<int> h_dp(dpSize, 0);

    // Initialize first row and column on the HOST before copying to GPU.
    // This avoids a separate initialization kernel and keeps code simple.
    for (int j = 0; j <= n; ++j) h_dp[0 * (n + 1) + j] = j * GAP;
    for (int i = 0; i <= m; ++i) h_dp[i * (n + 1) + 0] = i * GAP;

    // ── Device pointers ───────────────────────────────────────────────────────
    int  *d_dp   = nullptr;
    char *d_seq1 = nullptr, *d_seq2 = nullptr;

    // ── CUDA Events for accurate timing ──────────────────────────────────────
    // We use two pairs of events:
    //   (startTotal, stopTotal) : wraps everything including memory transfers
    //   (startKernel, stopKernel) : wraps kernel launches only
    cudaEvent_t startTotal, stopTotal, startKernel, stopKernel;
    CUDA_CHECK(cudaEventCreate(&startTotal));
    CUDA_CHECK(cudaEventCreate(&stopTotal));
    CUDA_CHECK(cudaEventCreate(&startKernel));
    CUDA_CHECK(cudaEventCreate(&stopKernel));

    // ── Record START of total GPU time ────────────────────────────────────────
    CUDA_CHECK(cudaEventRecord(startTotal));

    // ── Allocate device memory ────────────────────────────────────────────────
    CUDA_CHECK(cudaMalloc((void**)&d_dp,   dpSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_seq1, m * sizeof(char)));
    CUDA_CHECK(cudaMalloc((void**)&d_seq2, n * sizeof(char)));

    // ── Copy data Host → Device ───────────────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(d_dp,   h_dp.data(),     dpSize * sizeof(int),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq1, seq1.c_str(),    m * sizeof(char),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq2, seq2.c_str(),    n * sizeof(char),      cudaMemcpyHostToDevice));

    // ── Kernel launch configuration ───────────────────────────────────────────
    // We use 256 threads per block (a common sweet-spot for most NVIDIA GPUs).
    // Grid size is computed per diagonal based on the number of active cells.
    const int THREADS_PER_BLOCK = 256;

    // ── Record START of kernel-only time ──────────────────────────────────────
    CUDA_CHECK(cudaEventRecord(startKernel));

    // ── MAIN WAVEFRONT LOOP ───────────────────────────────────────────────────
    // Anti-diagonal index 'diag' runs from 2 (cell [1][1]) to m+n (cell [m][n]).
    // Each iteration of this loop dispatches one CUDA kernel for one diagonal.
    // The host calls cudaDeviceSynchronize() between diagonals to enforce the
    // wavefront ordering constraint (diagonal d must finish before d+1 starts).
    for (int diag = 2; diag <= m + n; ++diag) {
        // Number of active cells on this anti-diagonal
        int i_start  = max(1, diag - n);
        int i_end    = min(m, diag - 1);
        int numCells = i_end - i_start + 1;

        if (numCells <= 0) continue;   // no valid cells (shouldn't happen for valid seqs)

        // Grid = ceil(numCells / THREADS_PER_BLOCK)
        int gridSize = (numCells + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        // Launch the kernel for this anti-diagonal.
        // Each thread computes exactly one DP cell.
        nwKernel<<<gridSize, THREADS_PER_BLOCK>>>(
            d_dp, d_seq1, d_seq2, m, n, diag,
            MATCH, MISMATCH, GAP
        );

        // Wait for all threads of this diagonal to finish before
        // proceeding to the next diagonal (wavefront synchronization).
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ── Record STOP of kernel-only time ──────────────────────────────────────
    CUDA_CHECK(cudaEventRecord(stopKernel));
    CUDA_CHECK(cudaEventSynchronize(stopKernel));
    CUDA_CHECK(cudaEventElapsedTime(&kernelMs, startKernel, stopKernel));

    // ── Copy result Device → Host ─────────────────────────────────────────────
    // Only the final cell [m][n] is needed, but we copy back the full matrix
    // for validation convenience (can be optimised to single int copy).
    CUDA_CHECK(cudaMemcpy(h_dp.data(), d_dp, dpSize * sizeof(int), cudaMemcpyDeviceToHost));

    // ── Record STOP of total GPU time ────────────────────────────────────────
    CUDA_CHECK(cudaEventRecord(stopTotal));
    CUDA_CHECK(cudaEventSynchronize(stopTotal));
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, startTotal, stopTotal));

    // ── Extract alignment score ───────────────────────────────────────────────
    int score = h_dp[m * (n + 1) + n];

    // ── Clean up device memory ────────────────────────────────────────────────
    CUDA_CHECK(cudaFree(d_dp));
    CUDA_CHECK(cudaFree(d_seq1));
    CUDA_CHECK(cudaFree(d_seq2));

    // Destroy CUDA events
    CUDA_CHECK(cudaEventDestroy(startTotal));
    CUDA_CHECK(cudaEventDestroy(stopTotal));
    CUDA_CHECK(cudaEventDestroy(startKernel));
    CUDA_CHECK(cudaEventDestroy(stopKernel));

    return score;
}

// =============================================================================
//  ALIGNMENT RESULT PRINTER
// =============================================================================
static void printResults(int seqLen, int cpuScore, int gpuScore,
                         double cpuMs, float gpuKernelMs, float gpuTotalMs)
{
    banner("DNA SEQUENCE ALIGNMENT USING CUDA");
    cout << fixed << setprecision(4);
    cout << "  Sequence Length   : " << seqLen         << "\n";
    cout << "\n  Scoring:\n";
    cout << "    Match    : " << MATCH    << "\n";
    cout << "    Mismatch : " << MISMATCH << "\n";
    cout << "    Gap      : " << GAP      << "\n";

    cout << "\n  CPU Alignment Score : " << cpuScore << "\n";
    cout << "  GPU Alignment Score : " << gpuScore << "\n";

    cout << "\n  CPU Time        : " << cpuMs       << " ms\n";
    cout << "  GPU Kernel Time : " << gpuKernelMs  << " ms\n";
    cout << "  GPU Total Time  : " << gpuTotalMs   << " ms\n";

    double speedupKernel = cpuMs / gpuKernelMs;
    double speedupTotal  = cpuMs / gpuTotalMs;
    cout << "\n  Speedup (kernel) : " << speedupKernel << "x\n";
    cout << "  Speedup (total)  : " << speedupTotal  << "x\n";

    bool match = (cpuScore == gpuScore);
    cout << "\n  Validation : " << (match ? "PASS ✓" : "FAIL ✗") << "\n";
    if (!match)
        cout << "  [WARNING] CPU and GPU scores differ!\n";
}

// =============================================================================
//  CSV EXPORT
//  Appends one row per benchmark run to results/benchmark.csv
// =============================================================================
static void appendCSV(const string& csvPath, int seqLen,
                      double cpuMs, float kernelMs, float totalMs,
                      int cpuScore, int gpuScore)
{
    bool fileExists = false;
    {
        ifstream f(csvPath);
        fileExists = f.good();
    }

    ofstream csv(csvPath, ios::app);
    if (!fileExists)
        csv << "SeqLen,CPU_ms,GPUKernel_ms,GPUTotal_ms,SpeedupKernel,"
               "SpeedupTotal,CPU_Score,GPU_Score,Validation\n";

    double sk = cpuMs / kernelMs;
    double st = cpuMs / totalMs;
    csv << seqLen   << ","
        << fixed << setprecision(4)
        << cpuMs    << ","
        << kernelMs << ","
        << totalMs  << ","
        << sk       << ","
        << st       << ","
        << cpuScore << ","
        << gpuScore << ","
        << (cpuScore == gpuScore ? "PASS" : "FAIL") << "\n";

    cout << "  [CSV] Result appended to " << csvPath << "\n";
}

// =============================================================================
//  SINGLE RUN
//  Runs CPU + GPU alignment for given sequences, prints results, appends CSV.
// =============================================================================
static void runAlignment(const string& seq1, const string& seq2,
                         const string& csvPath, bool verbose = true)
{
    int seqLen = (int)seq1.size();  // (assumes equal length for display)

    // ── CPU timing ────────────────────────────────────────────────────────────
    auto cpuStart = chrono::high_resolution_clock::now();
    int  cpuScore = cpuNeedlemanWunsch(seq1, seq2);
    auto cpuEnd   = chrono::high_resolution_clock::now();
    double cpuMs  = chrono::duration<double, milli>(cpuEnd - cpuStart).count();

    // ── GPU timing ────────────────────────────────────────────────────────────
    float gpuKernelMs = 0.0f, gpuTotalMs = 0.0f;
    int   gpuScore    = cudaNeedlemanWunsch(seq1, seq2, gpuKernelMs, gpuTotalMs);

    if (verbose)
        printResults(seqLen, cpuScore, gpuScore, cpuMs, gpuKernelMs, gpuTotalMs);

    appendCSV(csvPath, seqLen, cpuMs, gpuKernelMs, gpuTotalMs, cpuScore, gpuScore);
}

// =============================================================================
//  BENCHMARK MODE
//  Runs alignment for multiple sequence lengths, multiple trials each.
// =============================================================================
static void runBenchmark(const string& csvPath) {
    banner("BENCHMARK MODE");

    // Sequence sizes to test. Adjust if GPU runs out of memory.
    vector<int> sizes = {500, 1000, 2000, 4000, 8000};
    const int   TRIALS = 3;   // average over this many runs

    for (int sz : sizes) {
        cout << "\n--- Sequence Length: " << sz << " ---\n";

        double totalCpuMs = 0;
        float  totalKernelMs = 0, totalGPUMs = 0;
        int    lastCpu = 0, lastGpu = 0;
        bool   allPass = true;

        for (int t = 0; t < TRIALS; ++t) {
            // Use different seeds per trial so each trial has different sequences
            string s1 = randomDNA(sz, 100 + t);
            string s2 = randomDNA(sz, 200 + t);

            // CPU
            auto cpuStart = chrono::high_resolution_clock::now();
            lastCpu       = cpuNeedlemanWunsch(s1, s2);
            auto cpuEnd   = chrono::high_resolution_clock::now();
            totalCpuMs   += chrono::duration<double, milli>(cpuEnd - cpuStart).count();

            // GPU
            float km = 0, gm = 0;
            lastGpu = cudaNeedlemanWunsch(s1, s2, km, gm);
            totalKernelMs += km;
            totalGPUMs    += gm;

            if (lastCpu != lastGpu) allPass = false;

            cout << "  Trial " << (t + 1) << ": CPU=" << lastCpu
                 << " GPU=" << lastGpu
                 << (lastCpu == lastGpu ? " PASS" : " FAIL") << "\n";
        }

        double avgCpu    = totalCpuMs    / TRIALS;
        float  avgKernel = totalKernelMs / TRIALS;
        float  avgTotal  = totalGPUMs    / TRIALS;

        cout << fixed << setprecision(4);
        cout << "  Avg CPU Time    : " << avgCpu    << " ms\n";
        cout << "  Avg Kernel Time : " << avgKernel << " ms\n";
        cout << "  Avg Total Time  : " << avgTotal  << " ms\n";
        cout << "  Speedup(kernel) : " << (avgCpu / avgKernel) << "x\n";
        cout << "  Speedup(total)  : " << (avgCpu / avgTotal)  << "x\n";
        cout << "  Validation      : " << (allPass ? "ALL PASS" : "SOME FAIL") << "\n";

        // Write average to CSV
        appendCSV(csvPath, sz, avgCpu, avgKernel, avgTotal, lastCpu, lastGpu);
    }

    cout << "\n[Benchmark Complete]\n";
}

// =============================================================================
//  MANUAL ALIGNMENT MODE
//  User types two DNA sequences; program aligns and prints result.
// =============================================================================
static void manualMode(const string& csvPath) {
    banner("MANUAL SEQUENCE INPUT");
    string seq1, seq2;
    cout << "Enter Sequence 1 (DNA, e.g. ACGTACGT): ";
    cin >> seq1;
    cout << "Enter Sequence 2 (DNA, e.g. ACGTACGT): ";
    cin >> seq2;

    // Convert to uppercase
    for (auto& c : seq1) c = toupper(c);
    for (auto& c : seq2) c = toupper(c);

    runAlignment(seq1, seq2, csvPath, true);
}

// =============================================================================
//  RANDOM ALIGNMENT MODE
//  User picks length; program generates two random sequences and aligns them.
// =============================================================================
static void randomMode(const string& csvPath) {
    banner("RANDOM SEQUENCE GENERATION");
    int len;
    cout << "Enter sequence length: ";
    cin >> len;
    if (len <= 0 || len > 20000) {
        cout << "[ERROR] Length must be 1–20000.\n";
        return;
    }
    string s1 = randomDNA(len, 42);
    string s2 = randomDNA(len, 99);
    cout << "Generated Seq1 (first 60 chars): " << s1.substr(0, min(60, len)) << "...\n";
    cout << "Generated Seq2 (first 60 chars): " << s2.substr(0, min(60, len)) << "...\n";
    runAlignment(s1, s2, csvPath, true);
}

// =============================================================================
//  SCORING PARAMETER CONFIGURATION
// =============================================================================
static void configureScoring() {
    banner("SCORING CONFIGURATION");
    cout << "Current: Match=" << MATCH << "  Mismatch=" << MISMATCH << "  Gap=" << GAP << "\n";
    cout << "Enter new Match score    : "; cin >> MATCH;
    cout << "Enter new Mismatch score : "; cin >> MISMATCH;
    cout << "Enter new Gap score      : "; cin >> GAP;
    cout << "Scoring updated: Match=" << MATCH << "  Mismatch=" << MISMATCH << "  Gap=" << GAP << "\n";
}

// =============================================================================
//  QUICK VALIDATION TEST
//  Tests a tiny known case:
//    seq1 = "ACGT"  seq2 = "ACT"
//  Verifies CPU == GPU.
// =============================================================================
static void quickValidationTest(const string& csvPath) {
    banner("QUICK VALIDATION TEST");
    string seq1 = "ACGT";
    string seq2 = "ACT";
    cout << "seq1 = " << seq1 << "\nseq2 = " << seq2 << "\n\n";

    // CPU
    int cpuScore = cpuNeedlemanWunsch(seq1, seq2);

    // GPU
    float km = 0, gm = 0;
    int gpuScore = cudaNeedlemanWunsch(seq1, seq2, km, gm);

    cout << "CPU Score : " << cpuScore << "\n";
    cout << "GPU Score : " << gpuScore << "\n";

    if (cpuScore == gpuScore)
        cout << "Results MATCH  ✓\n";
    else
        cout << "Results DO NOT MATCH  ✗\n";

    appendCSV(csvPath, (int)seq1.size(), 0.0, km, gm, cpuScore, gpuScore);
}

// =============================================================================
//  MAIN — Entry point + Interactive Menu
// =============================================================================
int main(int argc, char* argv[])
{
    // ── Print GPU info at startup ─────────────────────────────────────────────
    banner("DNA SEQUENCE ALIGNMENT USING CUDA");
    cout << "  Course  : Parallel Computing (BCS702)  |  Sem VII  |  Sec D\n";
    cout << "  Project : Parallel DNA Alignment — CCA 1\n";
    cout << "  Algorithm: Needleman-Wunsch + CUDA Wavefront Parallelism\n";

    printGPUInfo();

    // ── Determine CSV output path ─────────────────────────────────────────────
    // Place results in ./results/ directory (create if absent)
    string csvPath = "results/benchmark.csv";

    // ── Command-line shortcut: benchmark ─────────────────────────────────────
    if (argc > 1 && string(argv[1]) == "--benchmark") {
        runBenchmark(csvPath);
        return 0;
    }
    if (argc > 1 && string(argv[1]) == "--validate") {
        quickValidationTest(csvPath);
        return 0;
    }

    // ── Interactive menu ──────────────────────────────────────────────────────
    int choice = -1;
    while (choice != 0) {
        cout << "\n────────────────────────────────────────\n";
        cout << " MENU\n";
        cout << "────────────────────────────────────────\n";
        cout << " 1. Manual sequence input\n";
        cout << " 2. Random sequence generation\n";
        cout << " 3. Quick validation test  (ACGT vs ACT)\n";
        cout << " 4. Benchmark mode  (500→8000)\n";
        cout << " 5. Configure scoring parameters\n";
        cout << " 6. Display GPU information\n";
        cout << " 0. Exit\n";
        cout << "────────────────────────────────────────\n";
        cout << "Enter choice: ";
        cin >> choice;

        switch (choice) {
            case 1: manualMode(csvPath);         break;
            case 2: randomMode(csvPath);         break;
            case 3: quickValidationTest(csvPath); break;
            case 4: runBenchmark(csvPath);       break;
            case 5: configureScoring();          break;
            case 6: printGPUInfo();              break;
            case 0: cout << "Exiting. Results saved to " << csvPath << "\n"; break;
            default: cout << "Invalid choice.\n";
        }
    }

    return 0;
}
