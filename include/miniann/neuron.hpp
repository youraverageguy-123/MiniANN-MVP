#pragma once
#include "miniann/types.hpp"
#include "miniann/activation.hpp"
#include <random>
#include <cstddef>

namespace miniann {

enum class WeightInit { Uniform, Xavier, He };

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
    // Output of the most recent forward() (re-fires cached pre-activation).
    // Used by live monitors; training math is unaffected (caches rewritten
    // on the next forward pass before any backward()).
    double lastOutput() const { return activation_->activate(lastZ_); }

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
    double delta_ = 0.0;
};

} // namespace miniann
