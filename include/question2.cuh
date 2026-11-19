#ifndef HPCA_QUESTION2_CUH
#define HPCA_QUESTION2_CUH

#include <string>
#include <vector>

#include "question1.cuh"

namespace question2 {

// Read the T,P,f columns from Question 1.
question1::Curve read_curve_csv(const std::string& filename);

// 2(a): recover theta on the GPU at t[k] = k * dt.
std::vector<double> recover_theta(const std::vector<double>& forward_rate,
                                  double dt, double a, double sigma);

// 2(b): price the call by GPU Monte Carlo using P = A * exp(-B * r).
// config.t_max is the option expiry. Use one drift coefficient per step.
double price_option(const question1::SimulationConfig& config,
                    double initial_rate,
                    const question1::TransitionCoefficients& coefficients,
                    double bond_a, double bond_b, double strike);

// Write one T,theta row per grid point.
void write_theta_csv(const std::string& filename,
                     const std::vector<double>& time,
                     const std::vector<double>& theta);

}  // namespace question2

#endif  // HPCA_QUESTION2_CUH
