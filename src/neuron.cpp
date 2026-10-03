#include "miniann/neuron.hpp"
#include <cmath>

namespace miniann {

Neuron::Neuron(std::size_t numInputs, std::size_t numOutputs, ActivationPtr act,
               std::mt19937& rng, WeightInit initMethod)
    : activation_(std::move(act)), lastInputs_(numInputs, 0.0) {
    if (!activation_) throw std::invalid_argument("Neuron: null activation");
    if (numInputs == 0) throw std::invalid_argument("Neuron: numInputs must be > 0");
    gradWeights_.assign(numInputs, 0.0);
    gradBias_ = 0.0;
    weights_.resize(numInputs);
    switch (initMethod) {
        case WeightInit::Uniform: {
            std::uniform_real_distribution<double> d(-1.0, 1.0);
            for (auto& w : weights_) w = d(rng);
            std::uniform_real_distribution<double> db(-0.5, 0.5);
            bias_ = db(rng);
            break;
        }
        case WeightInit::Xavier: {
            if (numInputs + numOutputs == 0)
                throw std::invalid_argument("Neuron: Xavier needs fanIn+fanOut > 0");
            double limit = std::sqrt(6.0 / double(numInputs + numOutputs));
            std::uniform_real_distribution<double> d(-limit, limit);
            for (auto& w : weights_) w = d(rng);
            bias_ = 0.0;
            break;
        }
        case WeightInit::He: {
            double std = numInputs > 0 ? std::sqrt(2.0 / double(numInputs)) : 1.0;
            std::normal_distribution<double> d(0.0, std);
            for (auto& w : weights_) w = d(rng);
            bias_ = 0.0;
            break;
        }
    }
}

double Neuron::forward(const Vector& inputs) {
    lastInputs_ = inputs;
    double z = bias_;
    for (std::size_t i = 0; i < weights_.size() && i < inputs.size(); ++i)
        z += weights_[i] * inputs[i];
    lastZ_ = z;
    hasCachedA_ = false;
    return activation_->activate(z);
}

double Neuron::forwardLogit(const Vector& inputs) {
    lastInputs_ = inputs;
    double z = bias_;
    for (std::size_t i = 0; i < weights_.size() && i < inputs.size(); ++i)
        z += weights_[i] * inputs[i];
    lastZ_ = z;
    hasCachedA_ = false;
    return z;
}

void Neuron::backwardLogit(double dLoss_dZ) {
    delta_ = dLoss_dZ;
    for (std::size_t i = 0; i < weights_.size() && i < lastInputs_.size(); ++i)
        gradWeights_[i] += delta_ * lastInputs_[i];
    gradBias_ += delta_;
}

void Neuron::backward(double dLoss_dOutput) {
    // delta = dL/da * f'(z); accumulate grads (+=) for mini-batch support.
    delta_ = dLoss_dOutput * activation_->derivative(lastZ_);
    for (std::size_t i = 0; i < weights_.size() && i < lastInputs_.size(); ++i)
        gradWeights_[i] += delta_ * lastInputs_[i];
    gradBias_ += delta_;
}

void Neuron::setParameters(Vector w, double b) {
    weights_ = std::move(w);
    bias_ = b;
    gradWeights_.assign(weights_.size(), 0.0);
    gradBias_ = 0.0;
}

void Neuron::applyStep(const Vector& dW, double dB) {
    for (std::size_t i = 0; i < weights_.size() && i < dW.size(); ++i)
        weights_[i] += dW[i];
    bias_ += dB;
}

void Neuron::zeroGradients() {
    for (auto& g : gradWeights_) g = 0.0;
    gradBias_ = 0.0;
    delta_ = 0.0;
}

} // namespace miniann
