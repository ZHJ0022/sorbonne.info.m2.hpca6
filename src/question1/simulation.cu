
#include "question1.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace question1 {

// ============================================================
// 1. Gestion des erreurs CUDA
// ============================================================

static void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(error)
        );
    }
}

// ============================================================
// 2. Kernel principal : simulation des trajectoires
// ============================================================

__global__ void simulate_paths_kernel(
    double* partial_discount,
    double* partial_rate_discount,
    const double* drift,
    std::uint64_t num_paths,
    int num_steps,
    double dt,
    double initial_rate,
    double decay,
    double noise_stddev,
    unsigned long long seed
) {
    // Identifiant global du thread
    const unsigned int tid = threadIdx.x;

    const std::uint64_t path_id =
        static_cast<std::uint64_t>(blockIdx.x) *
        blockDim.x + tid;

    const bool active = path_id < num_paths;

    // Mémoire partagée pour la réduction
    extern __shared__ double shared[];

    double* shared_discount = shared;
    double* shared_rate_discount = shared + blockDim.x;

    // Chaque thread possède son propre état aléatoire
    curandStatePhilox4_32_10_t rng;

    double r = initial_rate;
    double integral = 0.0;

    if (active) {
        curand_init(seed, path_id, 0ULL, &rng);
    }

    // Parcours de tous les instants, y compris t = 0
    for (int k = 0; k <= num_steps; ++k) {

        double discount = 0.0;
        double rate_discount = 0.0;

        if (active) {

            // Évolution du taux après l'instant initial
            if (k > 0) {
                const double G = curand_normal_double(&rng);

                const double r_next =
                    decay * r
                    + drift[k - 1]
                    + noise_stddev * G;

                // Intégration par la méthode des trapèzes
                integral += 0.5 * dt * (r + r_next);

                r = r_next;
            }

            // Facteur d'actualisation
            discount = exp(-integral);

            // Contribution de cette trajectoire
            rate_discount = r * discount;
        }

        // Chaque thread écrit sa contribution
        shared_discount[tid] = discount;
        shared_rate_discount[tid] = rate_discount;

        __syncthreads();

        // Réduction parallèle au sein du bloc
        for (unsigned int stride = blockDim.x / 2;
             stride > 0;
             stride >>= 1) {

            if (tid < stride) {
                shared_discount[tid] +=
                    shared_discount[tid + stride];

                shared_rate_discount[tid] +=
                    shared_rate_discount[tid + stride];
            }

            __syncthreads();
        }

        // Le thread 0 enregistre les sommes du bloc
        if (tid == 0) {
            const std::size_t index =
                static_cast<std::size_t>(blockIdx.x) *
                (num_steps + 1) + k;

            partial_discount[index] = shared_discount[0];

            partial_rate_discount[index] =
                shared_rate_discount[0];
        }

        // Tous les threads doivent terminer avant
        // de réutiliser la mémoire partagée
        __syncthreads();
    }
}

// ============================================================
// 3. Kernel final : réduction entre les blocs
// ============================================================

__global__ void finalize_sums_kernel(
    const double* partial_discount,
    const double* partial_rate_discount,
    double* sum_discount,
    double* sum_rate_discount,
    std::uint64_t num_blocks,
    int num_points
) {
    const int k =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= num_points) {
        return;
    }

    double S = 0.0;
    double Q = 0.0;

    // Chaque thread calcule une échéance
    for (std::uint64_t b = 0; b < num_blocks; ++b) {

        const std::size_t index =
            static_cast<std::size_t>(b) * num_points + k;

        S += partial_discount[index];
        Q += partial_rate_discount[index];
    }

    sum_discount[k] = S;
    sum_rate_discount[k] = Q;
}

// ============================================================
// 4. Fonction principale appelée par main.cu
// ============================================================

SimulationSums simulate(
    const SimulationConfig& config,
    double initial_rate,
    const TransitionCoefficients& coefficients
) {

    // --------------------------------------------------------
    // Validation des paramètres
    // --------------------------------------------------------

    if (config.num_paths == 0 ||
        config.num_steps <= 0 ||
        !std::isfinite(config.t_max) ||
        config.t_max <= 0.0 ||
        config.t_max > 10.0) {
        throw std::invalid_argument("Invalid simulation configuration");
    }

    if (!std::isfinite(initial_rate)) {
        throw std::invalid_argument("Invalid initial rate");
    }

    if (!std::isfinite(coefficients.decay) ||
        !std::isfinite(coefficients.noise_stddev) ||
        coefficients.noise_stddev < 0.0 ||
        coefficients.drift.size() !=
            static_cast<std::size_t>(config.num_steps)) {
        throw std::invalid_argument("Invalid transition coefficients");
    }

    for (double value : coefficients.drift) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Non-finite drift coefficient");
        }
    }

    const int threads = config.threads_per_block;

    if (threads <= 0 || threads > 1024 ||
        (threads & (threads - 1)) != 0) {
        throw std::invalid_argument(
            "threads_per_block must be a power of two <= 1024"
        );
    }

    // --------------------------------------------------------
    // Dimensions de la grille CUDA
    // --------------------------------------------------------

    const std::uint64_t num_blocks =
        (config.num_paths - 1) / threads + 1;

    const int num_points = config.num_steps + 1;

    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");

    cudaDeviceProp properties;
    check_cuda(
        cudaGetDeviceProperties(&properties, device),
        "cudaGetDeviceProperties"
    );

    if (threads > properties.maxThreadsPerBlock ||
        2ULL * threads * sizeof(double) >
            properties.sharedMemPerBlock ||
        num_blocks >
            static_cast<std::uint64_t>(properties.maxGridSize[0])) {
        throw std::invalid_argument(
            "Unsupported CUDA grid or block configuration"
        );
    }

    const double dt = config.t_max / config.num_steps;

    // Vérification de la taille des allocations
    const std::size_t points =
        static_cast<std::size_t>(num_points);

    if (num_blocks >
        std::numeric_limits<std::size_t>::max() /
        points / sizeof(double)) {
        throw std::overflow_error("GPU allocation size overflow");
    }

    const std::size_t partial_size =
        static_cast<std::size_t>(num_blocks) * points;

    // --------------------------------------------------------
    // Allocation mémoire GPU
    // --------------------------------------------------------

    double* d_drift = nullptr;
    double* d_partial_discount = nullptr;
    double* d_partial_rate_discount = nullptr;
    double* d_sum_discount = nullptr;
    double* d_sum_rate_discount = nullptr;

    try {

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&d_drift),
                config.num_steps * sizeof(double)
            ),
            "cudaMalloc drift"
        );

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&d_partial_discount),
                partial_size * sizeof(double)
            ),
            "cudaMalloc partial discount"
        );

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&d_partial_rate_discount),
                partial_size * sizeof(double)
            ),
            "cudaMalloc partial rate discount"
        );

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&d_sum_discount),
                points * sizeof(double)
            ),
            "cudaMalloc sum discount"
        );

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&d_sum_rate_discount),
                points * sizeof(double)
            ),
            "cudaMalloc sum rate discount"
        );

        // ----------------------------------------------------
        // Transfert des coefficients CPU vers GPU
        // ----------------------------------------------------

        check_cuda(
            cudaMemcpy(
                d_drift,
                coefficients.drift.data(),
                config.num_steps * sizeof(double),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy drift"
        );

        // ----------------------------------------------------
        // Lancement du kernel de simulation
        // ----------------------------------------------------

        const std::size_t shared_bytes =
            2ULL * threads * sizeof(double);

        simulate_paths_kernel<<<
            static_cast<unsigned int>(num_blocks),
            threads,
            shared_bytes
        >>>(
            d_partial_discount,
            d_partial_rate_discount,
            d_drift,
            config.num_paths,
            config.num_steps,
            dt,
            initial_rate,
            coefficients.decay,
            coefficients.noise_stddev,
            config.seed
        );

        check_cuda(cudaGetLastError(), "simulate_paths_kernel");
        check_cuda(cudaDeviceSynchronize(), "simulation synchronization");

        // ----------------------------------------------------
        // Réduction finale des résultats
        // ----------------------------------------------------

        const int reduction_blocks =
            (num_points + threads - 1) / threads;

        finalize_sums_kernel<<<reduction_blocks, threads>>>(
            d_partial_discount,
            d_partial_rate_discount,
            d_sum_discount,
            d_sum_rate_discount,
            num_blocks,
            num_points
        );

        check_cuda(cudaGetLastError(), "finalize_sums_kernel");
        check_cuda(cudaDeviceSynchronize(), "reduction synchronization");

        // ----------------------------------------------------
        // Récupération des résultats sur CPU
        // ----------------------------------------------------

        SimulationSums result;

        result.sum_discount.resize(points);
        result.sum_rate_discount.resize(points);

        check_cuda(
            cudaMemcpy(
                result.sum_discount.data(),
                d_sum_discount,
                points * sizeof(double),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy sum discount"
        );

        check_cuda(
            cudaMemcpy(
                result.sum_rate_discount.data(),
                d_sum_rate_discount,
                points * sizeof(double),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy sum rate discount"
        );

        // ----------------------------------------------------
        // Libération de la mémoire GPU
        // ----------------------------------------------------

        check_cuda(cudaFree(d_drift), "cudaFree drift");
        d_drift = nullptr;

        check_cuda(cudaFree(d_partial_discount), "cudaFree partial discount");
        d_partial_discount = nullptr;

        check_cuda(cudaFree(d_partial_rate_discount), "cudaFree partial rate discount");
        d_partial_rate_discount = nullptr;

        check_cuda(cudaFree(d_sum_discount), "cudaFree sum discount");
        d_sum_discount = nullptr;

        check_cuda(cudaFree(d_sum_rate_discount), "cudaFree sum rate discount");
        d_sum_rate_discount = nullptr;

        return result;

    } catch (...) {

        // Nettoyage en cas d'erreur
        if (d_drift) cudaFree(d_drift);
        if (d_partial_discount) cudaFree(d_partial_discount);
        if (d_partial_rate_discount) cudaFree(d_partial_rate_discount);
        if (d_sum_discount) cudaFree(d_sum_discount);
        if (d_sum_rate_discount) cudaFree(d_sum_rate_discount);

        throw;
    }
}

} // namespace question1
