#include "cuda_check.cuh"
#include <vector>
#include <chrono>

// Exercise 5: grid-stride kernel. Same host program as vector_benchmark.cu
// (timed) so kernel times and effective bandwidth can be compared across
// launch configurations.
#ifndef N_VAL
#define N_VAL 1000003
#endif
#ifndef BLOCKS_VAL
#define BLOCKS_VAL 128
#endif
#ifndef THREADS_VAL
#define THREADS_VAL 256
#endif

__global__ void vector_add(const float *a, const float *b,
                           float *c, int n)
{
    int first = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = first; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    const int n = N_VAL;
    const int threads = THREADS_VAL;
    const int blocks = BLOCKS_VAL;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = (i % 97) * 0.25f;
        b[i] = (i % 31) * 0.5f;
    }

    using Clock = std::chrono::steady_clock;
    std::vector<float> reference(n);
    auto cpu_start = Clock::now();
    for (int i = 0; i < n; ++i) {
        reference[i] = a[i] + b[i];
    }
    double cpu_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - cpu_start).count();

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c), bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm up runtime and kernel before measuring.
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto wall_start = Clock::now();
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    double operation_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - wall_start).count();

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    double gbps = (12.0 * n) / (kernel_ms * 1e6); // 12n bytes / Tk
    std::printf("n=%d blocks=%d threads=%d CPU=%.6f ms kernel=%.6f ms "
                "operation=%.6f ms eff_bw=%.2f GB/s\n",
                n, blocks, threads, cpu_ms, kernel_ms, operation_ms, gbps);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    int errors = 0;
    for (int i = 0; i < n; ++i) {
        double expected = reference[i];
        if (!close_enough(c[i], expected)) ++errors;
    }
    std::printf("errors=%d\n", errors);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return errors != 0;
}
