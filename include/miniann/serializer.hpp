#pragma once
#include "miniann/network.hpp"
#include "miniann/trainer.hpp"
#include <string>
#include <random>

namespace miniann {

class ModelSerializer {
public:
    static void save(const NeuralNetwork& net, const std::string& path);
    static NeuralNetwork load(const std::string& path, std::mt19937& rng);
};

class CSVLossExporter {
public:
    static void exportHistory(const TrainingHistory& hist, const std::string& filepath);
};

} // namespace miniann
