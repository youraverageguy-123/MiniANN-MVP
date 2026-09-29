#pragma once
#include "miniann/neuron.hpp"

namespace miniann {

class Layer {
public:
    Layer(std::size_t numNeurons, std::size_t numInputs, ActivationPtr act,
          std::mt19937& rng, WeightInit init = WeightInit::Xavier);

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
