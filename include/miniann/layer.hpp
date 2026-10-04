#pragma once
#include "miniann/neuron.hpp"

namespace miniann {

class Layer {
public:
    // fanOut defaults to numNeurons (previous behavior). Pass the next-layer
    // width for a correct Xavier limit on hidden layers.
    Layer(std::size_t numNeurons, std::size_t numInputs, ActivationPtr act,
          std::mt19937& rng, WeightInit init = WeightInit::Xavier,
          std::size_t fanOut = 0);

    Vector forward(const Vector& inputs);
    Vector backward(const Vector& dLoss_dOutput); // returns dLoss/dInputs
    void zeroGradients();

    std::size_t size() const { return neurons_.size(); }
    std::size_t inputSize() const { return inputSize_; }
    std::vector<Neuron>& neurons() { return neurons_; }
    const std::vector<Neuron>& neurons() const { return neurons_; }

private:
    std::vector<Neuron> neurons_;
    std::size_t inputSize_;
};

} // namespace miniann
