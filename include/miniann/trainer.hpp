#pragma once
#include "miniann/network.hpp"
#include "miniann/dataset.hpp"
#include "miniann/loss.hpp"
#include "miniann/optimizer.hpp"
#include "miniann/logger.hpp"
#include <vector>
#include <cstddef>

namespace miniann {

struct TrainingConfig {
    int epochs = 1000;
    std::size_t batchSize = 1; // 1 = online SGD, >1 = mini-batch, 0 = full batch
    bool shuffle = true;
    unsigned seed = 42;
    int logEvery = 100;
};

struct TrainingHistory {
    std::vector<double> trainLoss;
    std::vector<double> validationLoss;
};

class Trainer {
public:
    Trainer(NeuralNetwork& net, const ILoss& loss, IOptimizer& opt,
            ILogger* logger = nullptr);
    TrainingHistory fit(const Dataset& train, const Dataset* validation,
                        const TrainingConfig& cfg);
private:
    NeuralNetwork& net_;
    const ILoss& loss_;
    IOptimizer& opt_;
    ILogger* logger_;
};

} // namespace miniann
