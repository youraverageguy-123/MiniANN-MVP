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
    virtual std::string name() const = 0; // runtime identity (viva: polymorphism)
};

class MSELoss : public ILoss {
public:
    double compute(const Vector& predicted, const Vector& target) const override;
    Vector gradient(const Vector& predicted, const Vector& target) const override;
    std::string name() const override { return "mse"; }
};

// Binary cross-entropy for single-output sigmoid nets. Predictions are
// clipped to [eps, 1-eps] so log() never touches 0.
class BCELoss : public ILoss {
public:
    explicit BCELoss(double eps = 1e-12) : eps_(eps) {}
    double compute(const Vector& predicted, const Vector& target) const override;
    Vector gradient(const Vector& predicted, const Vector& target) const override;
    std::string name() const override { return "bce"; }
private:
    double eps_;
};

// Categorical cross-entropy over one-hot targets with independent sigmoid
// outputs: mean over outputs of -(t*log(p) + (1-t)*log(1-p)). Unlike the
// softmax variant, wrong classes get an explicit push toward 0, which is what
// makes argmax evaluation work. (A true softmax head is not implemented.)
class CCELoss : public ILoss {
public:
    explicit CCELoss(double eps = 1e-12) : eps_(eps) {}
    double compute(const Vector& predicted, const Vector& target) const override;
    Vector gradient(const Vector& predicted, const Vector& target) const override;
    std::string name() const override { return "cce"; }
private:
    double eps_;
};

// "mse" | "bce" | "cce". Throws invalid_argument otherwise.
class LossFactory {
public:
    static std::unique_ptr<ILoss> create(const std::string& name);
};

} // namespace miniann
