#pragma once
#include "miniann/layer.hpp"

namespace miniann {

class NeuralNetwork {
public:
    void addLayer(Layer layer); // throws invalid_argument on dim mismatch
    Vector predict(const Vector& input); // non-const: caches z/activations
    void backward(const Vector& dLoss_dOutput);
    void zeroGradients();

    std::size_t numLayers() const { return layers_.size(); }
    std::vector<Layer>& layers() { return layers_; }
    const std::vector<Layer>& layers() const { return layers_; }

private:
    std::vector<Layer> layers_;
};

} // namespace miniann
