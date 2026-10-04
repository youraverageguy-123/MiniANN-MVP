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
    virtual bool isVector() const { return false; } // true for joint ops like softmax
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
    explicit LeakyReLU(double alpha = 0.01) : alpha_(alpha) {
        if (!(alpha > 0.0) || alpha >= 1.0)
            throw std::invalid_argument("LeakyReLU: alpha must be in (0,1)");
    }
    double activate(double z) const override { return z > 0.0 ? z : alpha_ * z; }
    double derivative(double z) const override { return z > 0.0 ? 1.0 : alpha_; }
    std::string name() const override { return "leaky_relu"; }
    double alpha() const { return alpha_; }
private:
    double alpha_;
};

class ELU : public IActivation {
public:
    explicit ELU(double alpha = 1.0) : alpha_(alpha) {
        if (alpha <= 0.0)
            throw std::invalid_argument("ELU: alpha must be > 0");
    }
    double activate(double z) const override {
        return z > 0.0 ? z : alpha_ * (std::exp(z) - 1.0);
    }
    double derivative(double z) const override {
        return z > 0.0 ? 1.0 : alpha_ * std::exp(z);
    }
    std::string name() const override { return "elu"; }
    double alpha() const { return alpha_; }
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

// Identity activation for regression output layers (Linear + MSE).
class Linear : public IActivation {
public:
    double activate(double z) const override { return z; }
    double derivative(double) const override { return 1.0; }
    std::string name() const override { return "linear"; }
};

// Joint softmax over a whole layer (stable z-max). Per-neuron activate()/
// derivative() throw: use Layer::forward/backward which applies the joint op.
class Softmax : public IActivation {
public:
    double activate(double) const override {
        throw std::logic_error("Softmax: use Layer joint forward, not per-neuron activate()");
    }
    double derivative(double) const override {
        throw std::logic_error("Softmax: use Layer joint backward, not per-neuron derivative()");
    }
    std::string name() const override { return "softmax"; }
    bool isVector() const override { return true; }
};

class ActivationFactory {
public:
    static ActivationPtr create(const std::string& name); // throws invalid_argument
};

} // namespace miniann
