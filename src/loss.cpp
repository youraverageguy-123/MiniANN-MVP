#include "miniann/loss.hpp"
#include <stdexcept>

namespace miniann {

double MSELoss::compute(const Vector& predicted, const Vector& target) const {
    if (predicted.size() != target.size())
        throw std::invalid_argument("MSELoss::compute: size mismatch");
    if (predicted.empty()) return 0.0;
    double s = 0.0;
    for (std::size_t i = 0; i < predicted.size(); ++i) {
        double d = predicted[i] - target[i];
        s += d * d;
    }
    return s / double(predicted.size());
}

Vector MSELoss::gradient(const Vector& predicted, const Vector& target) const {
    if (predicted.size() != target.size())
        throw std::invalid_argument("MSELoss::gradient: size mismatch");
    Vector g(predicted.size());
    double k = double(predicted.size());
    for (std::size_t i = 0; i < predicted.size(); ++i)
        g[i] = 2.0 * (predicted[i] - target[i]) / k;
    return g;
}

} // namespace miniann
