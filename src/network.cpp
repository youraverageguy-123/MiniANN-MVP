#include "miniann/network.hpp"
#include <stdexcept>

namespace miniann {

void NeuralNetwork::addLayer(Layer layer) {
    if (!layers_.empty()) {
        std::size_t prevOut = layers_.back().size();
        if (layer.inputSize() != prevOut)
            throw std::invalid_argument(
                "NeuralNetwork::addLayer: input size mismatch (expected " +
                std::to_string(prevOut) + ", got " + std::to_string(layer.inputSize()) + ")");
    }
    layers_.push_back(std::move(layer));
}

Vector NeuralNetwork::predict(const Vector& input) {
    if (layers_.empty()) throw std::runtime_error("NeuralNetwork::predict: no layers");
    Vector out = input;
    for (auto& l : layers_) out = l.forward(out);
    return out;
}

void NeuralNetwork::backward(const Vector& dLoss_dOutput) {
    Vector grad = dLoss_dOutput;
    for (int l = int(layers_.size()) - 1; l >= 0; --l)
        grad = layers_[std::size_t(l)].backward(grad);
}

void NeuralNetwork::zeroGradients() {
    for (auto& l : layers_) l.zeroGradients();
}

void NeuralNetwork::accept(INeuronVisitor& visitor) {
    for (std::size_t l = 0; l < layers_.size(); ++l)
        for (std::size_t j = 0; j < layers_[l].neurons().size(); ++j)
            visitor.visit(l, j, layers_[l].neurons()[j]);
}

} // namespace miniann
