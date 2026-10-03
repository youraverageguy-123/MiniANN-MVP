#include "miniann/neuron.hpp"
#include <cmath>

namespace miniann {

Neuron::Neuron(std::size_t numInputs, std::size_t numOutputs, ActivationPtr act,
               std::mt19937& rng, WeightInit initMethod)
    : activation_(std::move(act)), lastInputs_(numInputs, 0.0) {
    if (!activation_) throw std::invalid_argument("Neuron: null activation");
    gradWeights_.assign(numInputs, 0.0);
    gradBias_ = 0.0;
    weights_.resize(numInputs);
    WeightInitFactory::create(initMethod)->initialize(weights_, bias_, numInputs, numOutputs, rng);
}

void UniformInit::initialize(Vector& weights, double& bias, std::size_t,
                             std::size_t, std::mt19937& rng) const {
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (auto& w : weights) w = d(rng);
    std::uniform_real_distribution<double> db(-0.5, 0.5);
    bias = db(rng);
}

void XavierInit::initialize(Vector& weights, double& bias, std::size_t numInputs,
                             std::size_t numOutputs, std::mt19937& rng) const {
    double limit = std::sqrt(6.0 / double(numInputs + numOutputs));
    std::uniform_real_distribution<double> d(-limit, limit);
    for (auto& w : weights) w = d(rng);
    bias = 0.0;
}

void HeInit::initialize(Vector& weights, double& bias, std::size_t numInputs,
                        std::size_t, std::mt19937& rng) const {
    double std = numInputs > 0 ? std::sqrt(2.0 / double(numInputs)) : 1.0;
    std::normal_distribution<double> d(0.0, std);
    for (auto& w : weights) w = d(rng);
    bias = 0.0;
}

std::unique_ptr<IWeightInit> WeightInitFactory::create(WeightInit method) {
    switch (method) {
        case WeightInit::Uniform: return std::make_unique<UniformInit>();
        case WeightInit::Xavier:  return std::make_unique<XavierInit>();
        case WeightInit::He:      return std::make_unique<HeInit>();
    }
    throw std::invalid_argument("WeightInitFactory: unknown method");
}

std::unique_ptr<IWeightInit> WeightInitFactory::create(const std::string& name) {
    if (name == "uniform") return std::make_unique<UniformInit>();
    if (name == "xavier")  return std::make_unique<XavierInit>();
    if (name == "he")      return std::make_unique<HeInit>();
    throw std::invalid_argument("WeightInitFactory: unknown init '" + name +
                                "' (choose uniform|xavier|he)");
}

double Neuron::forward(const Vector& inputs) {
    lastInputs_ = inputs;
    double z = bias_;
    for (std::size_t i = 0; i < weights_.size() && i < inputs.size(); ++i)
        z += weights_[i] * inputs[i];
    lastZ_ = z;
    return activation_->activate(z);
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
