#include "question2.cuh"
#include "hull_white.cuh"

#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>

int main() {
    try {
        // Question 1 grid
        const auto curve = question2::read_curve_csv("question1.csv");
        const double dt = curve.time[1] - curve.time[0];

        // Recover theta (equation 10)
        const hull_white::Parameters model;
        const auto theta = question2::recover_theta(curve.forward_rate, dt,
                                                     model.a, model.sigma);

        std::filesystem::create_directories("results");
        const char* output = "results/question2_theta.csv";
        question2::write_theta_csv(output, curve.time, theta);

        // Compare with the two linear segments (equation 7)
        double sum_squared_error = 0.0;
        for (std::size_t k = 0; k < theta.size(); ++k) {
            const double error = theta[k] - hull_white::theta(curve.time[k]);
            sum_squared_error += error * error;
        }
        std::cout << "Theta RMSE: " << std::sqrt(sum_squared_error / theta.size())
                  << '\n' << "CSV: " << output << '\n';

        // 2(b): option expiry at the midpoint of the 0-10 year grid
        const int mid = static_cast<int>((curve.time.size() - 1) / 2);
        question1::SimulationConfig config;
        config.t_max = curve.time[mid];
        config.num_steps = mid;

        // Exact transition for linear interpolation of recovered theta
        question1::TransitionCoefficients coeff;
        coeff.decay = std::exp(-model.a * dt);
        coeff.noise_stddev = model.sigma *
            std::sqrt(-std::expm1(-2 * model.a * dt) / (2 * model.a));
        const double weight = -std::expm1(-model.a * dt) / model.a;
        coeff.drift.resize(config.num_steps);
        for (int k = 0; k < config.num_steps; ++k) {
            const double slope = (theta[k + 1] - theta[k]) / dt;
            coeff.drift[k] = theta[k] * weight + slope * (dt - weight) / model.a;
        }

        // Bond price at expiry: P(5,10) = A * exp(-B * r(5))
        const double expiry = config.t_max;
        const double maturity = curve.time.back();
        const double bond_b = -std::expm1(-model.a * (maturity - expiry)) / model.a;
        const double rate_variance = model.sigma * model.sigma *
            (-std::expm1(-2 * model.a * expiry)) / (2 * model.a);
        const double bond_a = curve.bond_price.back() / curve.bond_price[mid] *
            std::exp(bond_b * curve.forward_rate[mid] - 0.5 * rate_variance * bond_b * bond_b);
        const double strike = std::exp(-0.1);
        std::cout << "Option inputs: steps=" << config.num_steps << ", A=" << bond_a
                  << ", B=" << bond_b << ", K=" << strike << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
