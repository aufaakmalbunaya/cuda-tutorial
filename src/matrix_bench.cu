#include "cuda_check.cuh"
#include <vector>
#include <chrono>
#include <string>

// Timed square matrix multiplication for Exercise 7 tasks 2-3.
// Select kernel with -DUSE_TILED (default: naive). Size via -DN_VAL.
// Warm-up outside the timed interval; CUDA events around the kernel.
#ifndef N_VAL
#define N_VAL 256
#endif
constexpr int TILE = 16;

__global__ void matmul_naive(const float *a, const float *b, float *c,
                             int m, int k, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int q = 0; q < k; ++q) {
            sum += a[row * k + q] * b[q * n + col];
        }
        c[row * n + col] = sum;
    }
}

__global__ void matmul_tiled(const float *a, const float *b, float *c,
                             int m, int k, int n)
{
    __shared__ float a_tile[TILE][TILE];
    __shared__ float b_tile[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * TILE + tx;
    int row = blockIdx.y * TILE + ty;
    float sum = 0.0f;
    for (int base = 0; base < k; base += TILE) {
        int a_col = base + tx;
        int b_row = base + ty;
        a_tile[ty][tx] = (row < m && a_col < k)
            ? a[row * k + a_col] : 0.0f;
        b_tile[ty][tx] = (b_row < k && col < n)
            ? b[b_row * n + col] : 0.0f;
        __syncthreads();
        for (int q = 0; q < TILE; ++q) {
            sum += a_tile[ty][q] * b_tile[q][tx];
        }
        __syncthreads();
    }
    if (row < m && col < n) c[row * n + col] = sum;
}

int main()
{
    const int m = N_VAL, k = N_VAL, n = N_VAL;
#ifdef USE_TILED
    const char *variant = "tiled";
#else
    const char *variant = "naive";
#endif
    std::vector<float> a(m * k), b(k * n), c(m * n);
    for (int i = 0; i < m * k; ++i) a[i] = ((i % 17) - 8) * 0.125f;
    for (int i = 0; i < k * n; ++i) b[i] = ((i % 13) - 6) * 0.0625f;
    std::size_t a_bytes = a.size() * sizeof(float);
    std::size_t b_bytes = b.size() * sizeof(float);
    std::size_t c_bytes = c.size() * sizeof(float);
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a), a_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), b_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c), c_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), b_bytes, cudaMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x,
              (m + block.y - 1) / block.y);

    // Warm-up outside the timed interval.
#ifdef USE_TILED
    matmul_tiled<<<grid, block>>>(d_a, d_b, d_c, m, k, n);
#else
    matmul_naive<<<grid, block>>>(d_a, d_b, d_c, m, k, n);
#endif
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
#ifdef USE_TILED
    matmul_tiled<<<grid, block>>>(d_a, d_b, d_c, m, k, n);
#else
    matmul_naive<<<grid, block>>>(d_a, d_b, d_c, m, k, n);
#endif
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int idx = 0; idx < m * n; ++idx) {
        int row = idx / n, col = idx % n;
        double reference = 0.0;
        for (int q = 0; q < k; ++q) {
            reference += static_cast<double>(a[row * k + q]) * b[q * n + col];
        }
        if (!close_enough(c[idx], reference, 1e-4, 1e-4)) ++errors;
    }
    double gflops = (2.0 * m * k * n) / (kernel_ms * 1e6);
    std::printf("%s N=%d kernel=%.6f ms GFLOP/s=%.3f errors=%d\n",
                variant, n, kernel_ms, gflops, errors);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return errors != 0;
}
