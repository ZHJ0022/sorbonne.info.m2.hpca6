#ifndef HPCA_QUESTION1_CUH
#define HPCA_QUESTION1_CUH

#include <cstdint>
#include <string>
#include <vector>

namespace question1 {

// Run settings (model: hull_white.cuh); dt = t_max / num_steps.
// Require num_paths, num_steps > 0; finite t_max > 0; supported GPU block size.
struct SimulationConfig {
    std::uint64_t num_paths = 100000;
    int num_steps = 1000;
    double t_max = 10.0;
    std::uint64_t seed = 42;
    int threads_per_block = 256;
};

// CPU coefficients for k = 0, ..., num_steps - 1:
// r_next = decay*r + drift[k] + noise_stddev*G, G ~ N(0,1).
// decay = exp(-a*dt); noise_stddev = sigma*sqrt((1-exp(-2*a*dt))/(2*a)).
// drift[k] = integral[t_k,t_{k+1}] exp(-a*(t_{k+1}-u))*theta(u) du.
// Must match config/model; drift.size() == num_steps;
// all values finite, noise_stddev >= 0.
struct TransitionCoefficients {
    double decay = 0.0;
    double noise_stddev = 0.0;
    std::vector<double> drift;
};

// CPU raw sums, each of length num_steps + 1, including t_0.
// D = exp(-I); I_0 = 0; I_{k+1} = I_k + dt*(r_k+r_{k+1})/2.
// At t_0: sum_discount = num_paths,
//         sum_rate_discount = num_paths * initial_rate.
struct SimulationSums {
    std::vector<double> sum_discount;       // sum_i D_k^(i)
    std::vector<double> sum_rate_discount;  // sum_i r_k^(i)*D_k^(i)
};

// simulation.cu: CPU inputs; finite initial_rate = r(0).
// Manages GPU memory, transfers, RNG, simulation, reduction and cleanup.
// Returns completed CPU sums; throws on failure without returning partial results.
SimulationSums simulate(const SimulationConfig& config,
                        double initial_rate,
                        const TransitionCoefficients& coefficients);

// main.cu: CPU arrays of length num_steps + 1.
// time[k] = k*dt; bond_price[k] = S[k]/num_paths; forward_rate[k] = Q[k]/S[k],
// where S = sum_discount and Q = sum_rate_discount.
// At t_0: (time, bond_price, forward_rate) = (0, 1, initial_rate).
struct Curve {
    std::vector<double> time;
    std::vector<double> bond_price;
    std::vector<double> forward_rate;
};

// output.cpp: CSV header "T,P,f", one row per grid point, full double precision.
// Throws on unequal vector lengths or file errors.
void write_csv(const std::string& filename, const Curve& curve);

}  // namespace question1

#endif  // HPCA_QUESTION1_CUH
