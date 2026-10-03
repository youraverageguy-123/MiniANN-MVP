#pragma once
#include "miniann/network.hpp"
#include "miniann/trainer.hpp"
#include "miniann/optimizer.hpp"
#include <memory>
#include <string>
#include <random>

namespace miniann {

class ModelSerializer {
public:
    static void save(const NeuralNetwork& net, const std::string& path);
    // Save with optimizer state appended (MINIANN 2 opt block). nullptr = weights only.
    static void save(const NeuralNetwork& net, const IOptimizer* opt, const std::string& path);
    static NeuralNetwork load(const std::string& path, std::mt19937& rng);
    // Load weights + optional optimizer state (sets opt to nullptr when absent).
    static NeuralNetwork loadWithOptimizer(const std::string& path, std::mt19937& rng,
                                           std::unique_ptr<IOptimizer>& opt);
};

class CSVLossExporter {
public:
    static void exportHistory(const TrainingHistory& hist, const std::string& filepath);
};

} // namespace miniann
