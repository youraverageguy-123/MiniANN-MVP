#pragma once
#include "miniann/layer.hpp"

namespace miniann {

class NeuralNetwork;

// Visitor: an operation applied to every neuron in layer order. Optimizers
// implement this instead of reaching through layers()/neurons() accessors:
// the network owns the traversal, visitors own the per-neuron math
// (Tell-Don't-Ask / classic Visitor pattern).
class INeuronVisitor {
public:
    virtual ~INeuronVisitor() = default;
    virtual void visit(std::size_t layerIndex, std::size_t neuronIndex, Neuron& neuron) = 0;
};

class NeuralNetwork {
public:
    void addLayer(Layer layer); // throws invalid_argument on dim mismatch
    Vector predict(const Vector& input); // non-const: caches z/activations
    void backward(const Vector& dLoss_dOutput);
    void zeroGradients();
    void accept(INeuronVisitor& visitor); // in-order neuron traversal

    std::size_t numLayers() const { return layers_.size(); }
    std::vector<Layer>& layers() { return layers_; }
    const std::vector<Layer>& layers() const { return layers_; }

private:
    std::vector<Layer> layers_;
};

} // namespace miniann
