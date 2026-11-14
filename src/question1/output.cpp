#include "question1.cuh"

#include <cstddef>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace question1 {

void write_csv(const std::string& filename, const Curve& curve) {
    std::size_t n = curve.time.size();
    if (curve.bond_price.size() != n || curve.forward_rate.size() != n) {
        throw std::invalid_argument("curve arrays must have the same length");
    }

    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("cannot open CSV: " + filename);
    }

    file << "T,P,f\n";
    file << std::setprecision(17);

    // main.cu supplies N+1 ordered points, including T=0 and T_max.
    for (std::size_t k = 0; k < n; ++k) {
        file << curve.time[k] << ','
             << curve.bond_price[k] << ','
             << curve.forward_rate[k] << '\n';
    }

    file.close();
    if (!file) {
        throw std::runtime_error("cannot write CSV: " + filename);
    }
}

}  // namespace question1
