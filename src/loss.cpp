#include "miniann/loss.hpp"
#include <cmath>
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

static double clip01(double p, double eps) {
    if (p < eps) return eps;
    if (p > 1.0 - eps) return 1.0 - eps;
    return p;
}

double BCELoss::compute(const Vector& predicted, const Vector& target) const {
    if (predicted.size() != 1 || target.size() != 1)
        throw std::invalid_argument("BCELoss: binary loss needs 1 output (got " +
            std::to_string(predicted.size()) + ")");
    double p = clip01(predicted[0], eps_);
    double t = target[0];
    return -(t * std::log(p) + (1.0 - t) * std::log(1.0 - p));
}

Vector BCELoss::gradient(const Vector& predicted, const Vector& target) const {
    if (predicted.size() != 1 || target.size() != 1)
        throw std::invalid_argument("BCELoss: binary loss needs 1 output");
    double p = clip01(predicted[0], eps_);
    double t = target[0];
    return {(p - t) / (p * (1.0 - p))};
}

double CCELoss::compute(const Vector& predicted, const Vector& target) const {
    if (predicted.size() != target.size() || predicted.empty())
        throw std::invalid_argument("CCELoss::compute: size mismatch/empty");
    double s = 0.0;
    for (std::size_t i = 0; i < predicted.size(); ++i) {
        double p = clip01(predicted[i], eps_);
        double t = target[i];
        s -= t * std::log(p) + (1.0 - t) * std::log(1.0 - p);
    }
    return s / double(predicted.size());
}

Vector CCELoss::gradient(const Vector& predicted, const Vector& target) const {
    if (predicted.size() != target.size() || predicted.empty())
        throw std::invalid_argument("CCELoss::gradient: size mismatch/empty");
    Vector g(predicted.size());
    double k = double(predicted.size());
    for (std::size_t i = 0; i < predicted.size(); ++i) {
        double p = clip01(predicted[i], eps_);
        double t = target[i];
        g[i] = ((p - t) / (p * (1.0 - p))) / k;
    }
    return g;
}

std::unique_ptr<ILoss> LossFactory::create(const std::string& name) {
    if (name == "mse") return std::make_unique<MSELoss>();
    if (name == "bce") return std::make_unique<BCELoss>();
    if (name == "cce") return std::make_unique<CCELoss>();
    throw std::invalid_argument("LossFactory: unknown loss '" + name +
                                "' (choose mse|bce|cce)");
}

} // namespace miniann
