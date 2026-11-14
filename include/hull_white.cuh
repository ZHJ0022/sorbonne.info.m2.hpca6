#ifndef HPCA_HULL_WHITE_CUH
#define HPCA_HULL_WHITE_CUH

#include <cmath>
#include <stdexcept>

#include "question1.cuh"

namespace hull_white {

// CPU model parameters; negative r0 is allowed.
struct Parameters {
    double r0 = 0.012;
    double a = 1.0;
    double sigma = 0.1;
};

// Require finite r0, a, sigma; a > 0 and sigma >= 0.
inline void validate_parameters(const Parameters& p) {
    if (!std::isfinite(p.r0)) {
        throw std::invalid_argument("r0 must be finite");
    }
    if (!std::isfinite(p.a) || p.a <= 0.0) {
        throw std::invalid_argument("a must be finite and positive");
    }
    if (!std::isfinite(p.sigma) || p.sigma < 0.0) {
        throw std::invalid_argument("sigma must be finite and nonnegative");
    }
}

// Equation (7); require finite t in [0, 10].
inline double theta(double t) {
    if (!std::isfinite(t) || t < 0.0 || t > 10.0) {
        throw std::invalid_argument("theta requires finite t in [0, 10]");
    }
    if (t < 5.0) {
        return 0.012 + 0.0014 * t;
    }
    return 0.019 + 0.001 * (t - 5.0);
}

// Integral on one segment: theta(u) = theta(s) + slope*(u-s).
inline double linear_integral(double s, double t, double slope, double a) {
    double h = t - s;
    double A = -std::expm1(-a * h) / a;
    double B = (h - A) / a;
    return theta(s) * A + slope * B;
}

// CPU coefficients for equation (8); drift.size() == config.num_steps.
inline question1::TransitionCoefficients make_transition_coefficients(
    const question1::SimulationConfig& config, const Parameters& p) {
    validate_parameters(p);
    if (config.num_steps <= 0 || !std::isfinite(config.t_max) ||
        config.t_max <= 0.0 || config.t_max > 10.0) {
        throw std::invalid_argument("require num_steps > 0 and 0 < t_max <= 10");
    }

    double dt = config.t_max / config.num_steps;
    if (dt <= 0.0) {
        throw std::invalid_argument("dt is too small");
    }

    question1::TransitionCoefficients c;
    c.decay = std::exp(-p.a * dt);
    c.noise_stddev = p.sigma *
        std::sqrt(-std::expm1(-2.0 * p.a * dt) / (2.0 * p.a));
    if (!std::isfinite(c.decay) || !std::isfinite(c.noise_stddev) ||
        c.noise_stddev < 0.0) {
        throw std::runtime_error("invalid transition coefficients");
    }

    c.drift.resize(config.num_steps);
    for (int k = 0; k < config.num_steps; ++k) {
        double s = k * dt;
        double t = (k + 1) * dt;
        if (k == config.num_steps - 1) {
            t = config.t_max;
        }

        if (t <= 5.0) {
            c.drift[k] = linear_integral(s, t, 0.0014, p.a);
        } else if (s >= 5.0) {
            c.drift[k] = linear_integral(s, t, 0.001, p.a);
        } else {
            c.drift[k] = std::exp(-p.a * (t - 5.0)) *
                linear_integral(s, 5.0, 0.0014, p.a) +
                linear_integral(5.0, t, 0.001, p.a);
        }

        if (!std::isfinite(c.drift[k])) {
            throw std::runtime_error("non-finite drift coefficient");
        }
    }
    return c;
}

}  // namespace hull_white

#endif  // HPCA_HULL_WHITE_CUH