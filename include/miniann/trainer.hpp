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
    // Throws std::invalid_argument for settings that would crash or make no
    // sense (epochs < 1, logEvery < 1). Called by Trainer::fit().
    void validate() const;
};

struct TrainingHistory {
    std::vector<double> trainLoss;
    std::vector<double> validationLoss;
    std::vector<double> trainAcc;      // threshold/argmax accuracy per epoch
    std::vector<double> validationAcc;
};

// Streaming hook: called at the end of every epoch. Live monitors
// (console plots, GUIs) implement this; batch training is unaffected.
// shouldStop() is polled each epoch so a UI STOP button can cancel a run;
// the default (never stop) keeps all existing callbacks source-compatible.
//
// onEpochNet() is an optional live-visualization hook: the trainer calls it
// right after onEpoch() with read-only access to the current network so the
// UI can capture lightweight snapshots (weights/activations) at a controlled
// visualization interval. Training continues at full speed; the callback
// decides whether to copy (e.g. epoch % interval == 0). Default is a no-op,
// so all existing callbacks stay source-compatible.
class TrainingCallback {
public:
    virtual ~TrainingCallback() = default;
    virtual void onEpoch(int epoch, const TrainingHistory& hist) = 0;
    virtual void onEpochNet(int /*epoch*/, const NeuralNetwork& /*net*/) {}
    virtual bool shouldStop() const { return false; }
};

class Trainer {
public:
    Trainer(NeuralNetwork& net, const ILoss& loss, IOptimizer& opt,
            ILogger* logger = nullptr);
    TrainingHistory fit(const Dataset& train, const Dataset* validation,
                        const TrainingConfig& cfg, TrainingCallback* cb = nullptr);
private:
    NeuralNetwork& net_;
    const ILoss& loss_;
    IOptimizer& opt_;
    ILogger* logger_;
};

} // namespace miniann
