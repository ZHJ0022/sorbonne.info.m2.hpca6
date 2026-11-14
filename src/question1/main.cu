#include "question1.cuh"
#include "hull_white.cuh"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    try {
        // Use the default parameters defined in the headers.
        question1::SimulationConfig config;
        hull_white::Parameters model;
        std::string filename = "question1.csv";

        if (config.num_paths == 0 || config.threads_per_block <= 0) {
            throw std::invalid_argument("num_paths and threads_per_block must be positive");
        }
        // Also validates model parameters, num_steps and t_max.
        question1::TransitionCoefficients coefficients =
            hull_white::make_transition_coefficients(config, model);
        double dt = config.t_max / config.num_steps;
        double M = static_cast<double>(config.num_paths);

        // simulate() returns after GPU work and transfers have completed.
        std::chrono::steady_clock::time_point start =
            std::chrono::steady_clock::now();
        question1::SimulationSums sums =
            question1::simulate(config, model.r0, coefficients);
        std::chrono::steady_clock::time_point stop =
            std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = stop - start;

        // N updates give N+1 output points, including T=0.
        std::size_t num_points = static_cast<std::size_t>(config.num_steps) + 1;
        if (sums.sum_discount.size() != num_points ||
            sums.sum_rate_discount.size() != num_points) {
            throw std::runtime_error("simulation must return N+1 sums");
        }

        question1::Curve curve;
        curve.time.resize(num_points);
        curve.bond_price.resize(num_points);
        curve.forward_rate.resize(num_points);
        for (std::size_t k = 0; k < num_points; ++k) {
            double S = sums.sum_discount[k];
            double Q = sums.sum_rate_discount[k];
            if (!std::isfinite(S) || !std::isfinite(Q) || S <= 0.0) {
                throw std::runtime_error("invalid simulation sums");
            }
            curve.time[k] = k * dt;
            curve.bond_price[k] = S / M;
            curve.forward_rate[k] = Q / S;
            if (!std::isfinite(curve.bond_price[k]) ||
                !std::isfinite(curve.forward_rate[k])) {
                throw std::runtime_error("non-finite curve value");
            }
        }
        curve.time.back() = config.t_max;

        if (std::abs(curve.bond_price[0] - 1.0) > 1e-12 ||
            std::abs(curve.forward_rate[0] - model.r0) >
                1e-12 * (1.0 + std::abs(model.r0))) {
            throw std::runtime_error("expected P(0,0)=1 and f(0,0)=r0");
        }

        question1::write_csv(filename, curve);
        std::cout << "M=" << config.num_paths << ", N=" << config.num_steps
                  << ", T_max=" << config.t_max << ", dt=" << dt << '\n';
        std::cout << "r0=" << model.r0 << ", a=" << model.a
                  << ", sigma=" << model.sigma << '\n';
        std::cout << "seed=" << config.seed
                  << ", threads_per_block=" << config.threads_per_block << '\n';
        std::cout << "simulate() wall time: "
                  << elapsed.count() << " s\n";
        std::cout << "CSV: " << filename << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
