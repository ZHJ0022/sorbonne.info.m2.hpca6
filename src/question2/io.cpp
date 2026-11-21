#include "question2.cuh"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace question2 {

question1::Curve read_curve_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Cannot open " + filename);

    // Accept LF and CRLF headers.
    std::string line;
    std::getline(file, line);
    if (line != "T,P,f" && line != "T,P,f\r")
        throw std::runtime_error("Expected CSV header T,P,f");

    // Each row contains exactly T,P,f.
    question1::Curve curve;
    while (std::getline(file, line)) {
        std::istringstream row(line);
        double t, price, forward;
        char comma1, comma2, extra;
        if (!(row >> t >> comma1 >> price >> comma2 >> forward) ||
            comma1 != ',' || comma2 != ',' || (row >> extra))
            throw std::runtime_error("Invalid curve row: " + line);
        // Rates may be negative. Bond prices must be positive.
        if (!std::isfinite(t) || !std::isfinite(price) ||
            !std::isfinite(forward) || price <= 0)
            throw std::runtime_error("Invalid curve value: " + line);

        curve.time.push_back(t);
        curve.bond_price.push_back(price);
        curve.forward_rate.push_back(forward);
    }
    if (file.bad()) throw std::runtime_error("Cannot read " + filename);

    // An even number of steps puts year 5 on the grid.
    const std::size_t n = curve.time.size();
    if (n < 3 || (n - 1) % 2 != 0)
        throw std::runtime_error("Expected a grid containing years 0, 5 and 10");
    // Check each maturity against the uniform grid.
    const double dt = 10.0 / (n - 1);
    for (std::size_t k = 0; k < n; ++k) {
        if (std::abs(curve.time[k] - k * dt) > 1e-10)
            throw std::runtime_error("Expected a uniform grid from 0 to 10");
    }
    return curve;
}

void write_theta_csv(const std::string& filename,
                     const std::vector<double>& time,
                     const std::vector<double>& theta) {
    if (time.size() != theta.size())
        throw std::invalid_argument("Time and theta must have the same length");

    std::ofstream file(filename);
    if (!file) throw std::runtime_error("Cannot open " + filename);

    // Keep double precision in the CSV.
    file << "T,theta\n" << std::setprecision(17);
    for (std::size_t k = 0; k < time.size(); ++k)
        file << time[k] << ',' << theta[k] << '\n';

    file.close();
    if (!file) throw std::runtime_error("Cannot write " + filename);
}

}  // namespace question2
