#pragma once
#include "miniann/types.hpp"
#include "miniann/activation.hpp"
#include <random>
#include <cstddef>
#include <memory>
#include <string>

namespace miniann {

enum class WeightInit { Uniform, Xavier, He };

// Strategy: weight-initialization schemes as polymorphic objects, so Neuron
// never switches on an enum to decide math. New schemes subclass + register
// in the factory; no existing code changes.
class IWeightInit {
public:
    virtual ~IWeightInit() = default;
    virtual void initialize(Vector& weights, double& bias, std::size_t numInputs,
                            std::size_t numOutputs, std::mt19937& rng) const = 0;
    virtual std::string name() const = 0;
};

class UniformInit : public IWeightInit {
public:
    void initialize(Vector& weights, double& bias, std::size_t numInputs,
                    std::size_t numOutputs, std::mt19937& rng) const override;
    std::string name() const override { return "uniform"; }
};

class XavierInit : public IWeightInit {
public:
    void initialize(Vector& weights, double& bias, std::size_t numInputs,
                    std::size_t numOutputs, std::mt19937& rng) const override;
    std::string name() const override { return "xavier"; }
};

class HeInit : public IWeightInit {
public:
    void initialize(Vector& weights, double& bias, std::size_t numInputs,
                    std::size_t numOutputs, std::mt19937& rng) const override;
    std::string name() const override { return "he"; }
};

class WeightInitFactory {
public:
    static std::unique_ptr<IWeightInit> create(WeightInit method);
    static std::unique_ptr<IWeightInit> create(const std::string& name);
};

class Neuron {
public:
    Neuron(std::size_t numInputs, std::size_t numOutputs, ActivationPtr act,
           std::mt19937& rng, WeightInit initMethod = WeightInit::Xavier);

    double forward(const Vector& inputs);
    void backward(double dLoss_dOutput); // accumulates dL/dw, sets delta_

    double delta() const { return delta_; }
    const Vector& weights() const { return weights_; }
    double bias() const { return bias_; }
    const Vector& gradWeights() const { return gradWeights_; }
    double gradBias() const { return gradBias_; }
    const IActivation& activation() const { return *activation_; }
    double lastZ() const { return lastZ_; }
    // Raw logit z = b + w.x (caches inputs/z, clears softmax override).
    double forwardLogit(const Vector& inputs);
    // Cache a joint-activation output (softmax layer path).
    void setCachedOutput(double a) { lastA_ = a; hasCachedA_ = true; }
    // dL/dz directly (softmax Jacobian already applied by Layer).
    void backwardLogit(double dLoss_dZ);
    // Output of the most recent forward() (re-fires cached pre-activation).
    // Used by live monitors; training math is unaffected (caches rewritten
    // on the next forward pass before any backward()).
    double lastOutput() const {
        return hasCachedA_ ? lastA_ : activation_->activate(lastZ_);
    }

    void setParameters(Vector w, double b); // used by ModelSerializer
    void applyStep(const Vector& dW, double dB); // w += dW, b += dB (optimizers)
    void zeroGradients();

private:
    Vector weights_;
    double bias_;
    Vector gradWeights_;
    double gradBias_;
    ActivationPtr activation_;
    Vector lastInputs_;
    double lastZ_ = 0.0;
    double lastA_ = 0.0;
    bool hasCachedA_ = false;
    double delta_ = 0.0;
};

} // namespace miniann
