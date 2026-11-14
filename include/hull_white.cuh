#ifndef HPCA_HULL_WHITE_CUH
#define HPCA_HULL_WHITE_CUH

#include <cmath>
#include <stdexcept>

namespace hull_white {

// CPU model parameters; negative r0 is allowed.
struct Parameters {
    double r0 = 0.012;
    double a = 1.0;
    double sigma = 0.1;
};

// Call after constructing or changing parameters, before computing coefficients.
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

// Equation (7), defined for finite t in [0, 10]; both branches meet at t = 5.
inline double theta(double t) {
    if (!std::isfinite(t) || t < 0.0 || t > 10.0) {
        throw std::invalid_argument("theta requires finite t in [0, 10]");
    }
    return t < 5.0 ? 0.012 + 0.0014 * t
                   : 0.019 + 0.001 * (t - 5.0);
}

}  // namespace hull_white

#endif  // HPCA_HULL_WHITE_CUH
