#include "cuda_check.cuh"
#include <vector>
#include <chrono>

// Mini-project: 2D five-point stencil (one simulation step / image filter).
// y[r][c] = (x[r][c] + x[r-1][c] + x[r+1][c] + x[r][c-1] + x[r][c+1]) / 5
// Boundary cells are copied unchanged. Separate input/output arrays.
// Usage: ./stencil [H W steps]   (defaults: 257 259 10)

__global__ void stencil_step(const float *in, float *out, int h, int w)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= h || c >= w) return;
    int i = r * w + c;
    if (r == 0 || r == h - 1 || c == 0 || c == w - 1) {
        out[i] = in[i];
        return;
    }
    out[i] = (in[i] + in[i - w] + in[i + w] + in[i - 1] + in[i + 1]) * 0.2f;
}

static void stencil_cpu(const float *in, float *out, int h, int w)
{
    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            int i = r * w + c;
            if (r == 0 || r == h - 1 || c == 0 || c == w - 1) {
                out[i] = in[i];
            } else {
                out[i] = (in[i] + in[i - w] + in[i + w] +
                          in[i - 1] + in[i + 1]) * 0.2f;
            }
        }
    }
}

int main(int argc, char **argv)
{
    const int h = (argc > 1) ? std::atoi(argv[1]) : 257;
    const int w = (argc > 2) ? std::atoi(argv[2]) : 259;
    const int steps = (argc > 3) ? std::atoi(argv[3]) : 10;
    const std::size_t bytes = static_cast<std::size_t>(h) * w * sizeof(float);

    std::vector<float> in(h * w), ref(h * w), ref2(h * w), out(h * w);
    for (int i = 0; i < h * w; ++i)
        in[i] = ((i % 41) - 20) * 0.25f;

    // CPU reference: same number of steps, ping-pong on host.
    std::vector<float> ping = in, pong(h * w);
    auto cpu_start = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
        stencil_cpu(ping.data(), pong.data(), h, w);
        ping.swap(pong);
    }
    double cpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - cpu_start).count();
    ref = ping;

    float *d_a = nullptr, *d_b = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), bytes));
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x,
              (h + block.y - 1) / block.y);

    // Variant A: data stays on GPU; swap buffers between steps.
    CUDA_CHECK(cudaMemcpy(d_a, in.data(), bytes, cudaMemcpyHostToDevice));
    stencil_step<<<grid, block>>>(d_a, d_b, h, w); // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    auto wall_a = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(d_a, in.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    float *src = d_a, *dst = d_b;
    for (int s = 0; s < steps; ++s) {
        stencil_step<<<grid, block>>>(src, dst, h, w);
        CUDA_CHECK(cudaGetLastError());
        float *tmp = src; src = dst; dst = tmp;
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaMemcpy(out.data(), src, bytes, cudaMemcpyDeviceToHost));
    double opA_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wall_a).count();
    float kernA_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernA_ms, start, stop));

    int errors = 0;
    for (int i = 0; i < h * w; ++i)
        if (!close_enough(out[i], ref[i], 1e-4, 1e-4)) ++errors;

    // Variant B: H2D + kernel + D2H repeated for every step.
    auto wall_b = std::chrono::steady_clock::now();
    src = d_a; dst = d_b;
    for (int s = 0; s < steps; ++s) {
        CUDA_CHECK(cudaMemcpy(src, (s == 0 ? in.data() : out.data()),
                              bytes, cudaMemcpyHostToDevice));
        stencil_step<<<grid, block>>>(src, dst, h, w);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data(), dst, bytes, cudaMemcpyDeviceToHost));
        float *tmp = src; src = dst; dst = tmp;
    }
    double opB_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wall_b).count();
    int errors_b = 0;
    for (int i = 0; i < h * w; ++i)
        if (!close_enough(out[i], ref[i], 1e-4, 1e-4)) ++errors_b;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    std::printf("stencil H=%d W=%d steps=%d\n", h, w, steps);
    std::printf("CPU %d steps: %.3f ms\n", steps, cpu_ms);
    std::printf("GPU resident : kernel-all-steps=%.3f ms transfer-inclusive=%.3f ms errors=%d\n",
                kernA_ms, opA_ms, errors);
    std::printf("GPU per-step : transfer-inclusive=%.3f ms errors=%d\n", opB_ms, errors_b);
    return (errors != 0 || errors_b != 0);
}
