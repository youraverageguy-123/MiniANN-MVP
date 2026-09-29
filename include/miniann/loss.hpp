#pragma once
#include "miniann/types.hpp"
#include <cstddef>

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

} // namespace miniann
