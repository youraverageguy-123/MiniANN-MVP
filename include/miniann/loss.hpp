#pragma once
#include "miniann/types.hpp"
#include <cstddef>
#include <memory>
#include <string>

namespace miniann {

class ILoss {
public:
    virtual ~ILoss() = default;
    virtual double compute(const Vector& predicted, const Vector& target) const = 0;
    virtual Vector gradient(const Vector& predicted, const Vector& target) const = 0;
};

class MSELoss : public ILoss {
public:
    double compute(const Vector& predicted, const Vector& target) const override;
    Vector gradient(const Vector& predicted, const Vector& target) const override;
};

// Binary cross-entropy for single-output sigmoid nets. Predictions are
// clipped to [eps, 1-eps] so log() never touches 0.
class BCELoss : public ILoss {
public:
    explicit BCELoss(double eps = 1e-12) : eps_(eps) {}
    double compute(const Vector& predicted, const Vector& target) const override;
    Vector gradient(const Vector& predicted, const Vector& target) const override;
private:
    double eps_;
};

// Categorical cross-entropy over one-hot targets: -mean(t*log(p)).
// Pairs with a sigmoid/softmax-style multi-output head.
class CCELoss : public ILoss {
public:
    explicit CCELoss(double eps = 1e-12) : eps_(eps) {}
    double compute(const Vector& predicted, const Vector& target) const override;
    Vector gradient(const Vector& predicted, const Vector& target) const override;
private:
    double eps_;
};

// "mse" | "bce" | "cce". Throws invalid_argument otherwise.
class LossFactory {
public:
    static std::unique_ptr<ILoss> create(const std::string& name);
};

} // namespace miniann
