#include "miniann/layer.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace miniann {

Layer::Layer(std::size_t numNeurons, std::size_t numInputs, ActivationPtr act,
             std::mt19937& rng, WeightInit init, std::size_t fanOut)
    : inputSize_(numInputs) {
    if (numNeurons == 0) throw std::invalid_argument("Layer: numNeurons must be > 0");
    if (numInputs == 0) throw std::invalid_argument("Layer: numInputs must be > 0");
    if (fanOut == 0) fanOut = numNeurons;
    neurons_.reserve(numNeurons);
    for (std::size_t j = 0; j < numNeurons; ++j)
        neurons_.emplace_back(numInputs, fanOut, act, rng, init);
}

Vector Layer::forward(const Vector& inputs) {
    if (!neurons_.empty() && neurons_[0].activation().isVector()) {
        // Joint stable softmax over the whole layer.
        Vector logits;
        logits.reserve(neurons_.size());
        for (auto& n : neurons_) logits.push_back(n.forwardLogit(inputs));
        double m = *std::max_element(logits.begin(), logits.end());
        Vector out;
        out.reserve(logits.size());
        double sum = 0.0;
        for (double z : logits) {
            double e = std::exp(z - m);
            out.push_back(e);
            sum += e;
        }
        for (std::size_t j = 0; j < out.size(); ++j) {
            out[j] /= sum;
            neurons_[j].setCachedOutput(out[j]);
        }
        return out;
    }
    Vector out;
    out.reserve(neurons_.size());
    for (auto& n : neurons_) out.push_back(n.forward(inputs));
    return out;
}

Vector Layer::backward(const Vector& dLoss_dOutput) {
    if (dLoss_dOutput.size() != neurons_.size())
        throw std::invalid_argument("Layer::backward: size mismatch");
    if (!neurons_.empty() && neurons_[0].activation().isVector()) {
        // Softmax Jacobian: dz_i = p_i * (g_i - dot(g, p)).
        Vector probs;
        probs.reserve(neurons_.size());
        for (auto& n : neurons_) probs.push_back(n.lastOutput());
        double dot = 0.0;
        for (std::size_t j = 0; j < probs.size(); ++j) dot += dLoss_dOutput[j] * probs[j];
        for (std::size_t j = 0; j < neurons_.size(); ++j)
            neurons_[j].backwardLogit(probs[j] * (dLoss_dOutput[j] - dot));
        Vector dLoss_dInputs(inputSize_, 0.0);
        for (std::size_t j = 0; j < neurons_.size(); ++j) {
            const Vector& w = neurons_[j].weights();
            double d = neurons_[j].delta();
            for (std::size_t i = 0; i < inputSize_ && i < w.size(); ++i)
                dLoss_dInputs[i] += w[i] * d;
        }
        return dLoss_dInputs;
    }
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
