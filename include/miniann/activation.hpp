#pragma once
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <stdexcept>

namespace miniann {

class IActivation {
public:
    virtual ~IActivation() = default;
    virtual double activate(double z) const = 0;
    virtual double derivative(double z) const = 0; // evaluated at pre-activation z
    virtual std::string name() const = 0;
};

using ActivationPtr = std::shared_ptr<const IActivation>;

class Sigmoid : public IActivation {
public:
    double activate(double z) const override { return 1.0 / (1.0 + std::exp(-z)); }
    double derivative(double z) const override {
        double a = activate(z);
        return a * (1.0 - a);
    }
    std::string name() const override { return "sigmoid"; }
};

class Tanh : public IActivation {
public:
    double activate(double z) const override { return std::tanh(z); }
    double derivative(double z) const override {
        double a = std::tanh(z);
        return 1.0 - (a * a);
    }
    std::string name() const override { return "tanh"; }
};

class ReLU : public IActivation {
public:
    double activate(double z) const override { return z > 0.0 ? z : 0.0; }
    double derivative(double z) const override { return z > 0.0 ? 1.0 : 0.0; }
    std::string name() const override { return "relu"; }
};

class LeakyReLU : public IActivation {
public:
    explicit LeakyReLU(double alpha = 0.01) : alpha_(alpha) {}
    double activate(double z) const override { return z > 0.0 ? z : alpha_ * z; }
    double derivative(double z) const override { return z > 0.0 ? 1.0 : alpha_; }
    std::string name() const override { return "leaky_relu"; }
private:
    double alpha_;
};

class Swish : public IActivation {
public:
    double activate(double z) const override {
        return z / (1.0 + std::exp(-z)); // z * sigmoid(z)
    }
    double derivative(double z) const override {
        double s = 1.0 / (1.0 + std::exp(-z));
        return s + z * s * (1.0 - s); // sigmoid + z*sigmoid*(1-sigmoid)
    }
    std::string name() const override { return "swish"; }
};

class ActivationFactory {
public:
    static ActivationPtr create(const std::string& name); // throws invalid_argument
};

} // namespace miniann
