#include "miniann/layer.hpp"
#include <stdexcept>

namespace miniann {

Layer::Layer(std::size_t numNeurons, std::size_t numInputs, ActivationPtr act,
             std::mt19937& rng, WeightInit init)
    : inputSize_(numInputs) {
    if (numNeurons == 0) throw std::invalid_argument("Layer: numNeurons must be > 0");
    neurons_.reserve(numNeurons);
    for (std::size_t j = 0; j < numNeurons; ++j)
        neurons_.emplace_back(numInputs, numNeurons, act, rng, init);
}

Vector Layer::forward(const Vector& inputs) {
    Vector out;
    out.reserve(neurons_.size());
    for (auto& n : neurons_) out.push_back(n.forward(inputs));
    return out;
}

Vector Layer::backward(const Vector& dLoss_dOutput) {
    if (dLoss_dOutput.size() != neurons_.size())
        throw std::invalid_argument("Layer::backward: size mismatch");
    // 1. per-neuron backward (sets delta, accumulates grads)
    for (std::size_t j = 0; j < neurons_.size(); ++j)
        neurons_[j].backward(dLoss_dOutput[j]);
    // 2. propagate error down using UNMODIFIED weights:
    // dL/da_prev[i] = sum_j w_ji * delta_j
    Vector dLoss_dInputs(inputSize_, 0.0);
    for (std::size_t j = 0; j < neurons_.size(); ++j) {
        const Vector& w = neurons_[j].weights();
        double d = neurons_[j].delta();
        for (std::size_t i = 0; i < inputSize_ && i < w.size(); ++i)
            dLoss_dInputs[i] += w[i] * d;
    }
    return dLoss_dInputs;
}

void Layer::zeroGradients() {
    for (auto& n : neurons_) n.zeroGradients();
}

} // namespace miniann
