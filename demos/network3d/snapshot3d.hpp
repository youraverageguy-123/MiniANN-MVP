#pragma once
// Lightweight network snapshot for the 3D viewer (mirrors the GUI's
// VisualizationState field-for-field, but decoupled from gui_qt.cpp so the
// widget never depends on the main-window translation unit).
// Moc-free: plain struct, no Qt meta-object usage.
#include <cstddef>
#include <string>
#include <vector>
#include "miniann/types.hpp"

namespace net3d {

struct NetSnapshot3D {
    int epoch = 0;
    std::size_t inDim = 0;
    // weights[l][neuron][input], biases[l][neuron], one act name per layer
    // (hidden layers + output layer, in order).
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
    std::vector<std::string> acts;
    miniann::Vector probe;
};

} // namespace net3d
