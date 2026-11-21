#include "question2.cuh"

#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <stdexcept>

namespace question2 {

static void check_cuda(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

static __global__ void theta_kernel(const double* f, double* theta, int n,
                                    double dt, double a, double sigma) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;

    // One-sided differences at the ends, centered inside.
    double df;
    if (k == 0)
        df = (f[1] - f[0]) / dt;
    else if (k == n - 1)
        df = (f[k] - f[k - 1]) / dt;
    else
        df = (f[k + 1] - f[k - 1]) / (2 * dt);

    // Hull-White, equation (10).
    const double t = k * dt;
    theta[k] = df + a * f[k] + sigma * sigma * (-expm1(-2 * a * t)) / (2 * a);
}

std::vector<double> recover_theta(const std::vector<double>& forward_rate,
                                  double dt, double a, double sigma) {
    if (forward_rate.size() < 2 || forward_rate.size() > INT_MAX)
        throw std::invalid_argument("Invalid forward curve size");
    if (!std::isfinite(dt) || dt <= 0 || !std::isfinite(a) || a <= 0 ||
        !std::isfinite(sigma) || sigma < 0)
        throw std::invalid_argument("Invalid Hull-White parameters or time step");

    const int n = static_cast<int>(forward_rate.size());
    const std::size_t bytes = forward_rate.size() * sizeof(double);
    std::vector<double> theta(n);

    // GPU buffer: f followed by theta.
    double* d_f = nullptr;
    check_cuda(cudaMalloc(&d_f, 2 * bytes));
    double* d_theta = d_f + n;
    try {
        check_cuda(cudaMemcpy(d_f, forward_rate.data(), bytes, cudaMemcpyHostToDevice));
        const int threads = 256;
        const int blocks = 1 + (n - 1) / threads;
        theta_kernel<<<blocks, threads>>>(d_f, d_theta, n, dt, a, sigma);
        check_cuda(cudaGetLastError());
        check_cuda(cudaMemcpy(theta.data(), d_theta, bytes, cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(d_f);
        throw;
    }
    check_cuda(cudaFree(d_f));
    return theta;
}

}  // namespace question2
