#pragma once
// ExperimentController (§36 of the workbench spec): the single seam between
// the UI and the ANN backend.
//
//   UI -> ExperimentConfig -> ExperimentController -> ANN backend
//                                                        -> ExperimentResult -> UI
//
// The UI never constructs datasets, networks, losses or optimizers itself;
// it fills one ExperimentConfig and hands it over. Validation (§24) happens
// before any training object exists, so misconfiguration surfaces as a
// readable message instead of an exception deep in the trainer.
#include "miniann/network.hpp"
#include "miniann/dataset.hpp"
#include "miniann/loss.hpp"
#include "miniann/optimizer.hpp"
#include "miniann/trainer.hpp"
#include "miniann/metrics.hpp"
#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace miniann {

struct ExperimentConfig {
    // -- dataset --
    std::string dataset = "xor"; // and|or|xor|iris|csv
    std::string csvPath;
    int targetCol = -1;          // -1 = auto (last column)
    bool header = true;
    int normMode = 1;            // 0 none, 1 max-abs, 2 min-max
    double trainFrac = 0.8, valFrac = 0.1;
    unsigned seed = 42;
    bool shuffle = true;      // reshuffle the train set each epoch
    bool splitShuffle = true; // shuffle before the train/val/test split
    // -- network --
    std::vector<int> hidden = {8, 8};
    std::vector<std::string> hiddenActs = {"tanh", "tanh"};
    std::string outputAct = "sigmoid";
    // -- training --
    std::string loss = "mse";
    std::string optimizer = "adam";
    OptimizerConfig opt;
    std::string lrScheduler = "Constant";
    double weightDecay = 0.0;
    double gradClip = 0.0;
    int epochs = 1500;
    std::size_t batchSize = 1;   // 0 = full batch
};

// Dataset side of an experiment after loading, label encoding,
// deterministic splitting and normalization.
struct PreparedData {
    Dataset train, val, test;
    bool hasVal = false;
    bool hasTest = false;      // false only for tiny truth tables (no held-out data exists)
    bool tiny = false;           // <=8 samples: train on the full table, no val/test splits
    std::string splitNote;       // human-readable split description / adjustments
    std::string name;
    std::size_t samples = 0, features = 0, inDim = 0, outDim = 0;
    std::size_t classes = 0;     // >0 only when targets are discrete classes
    bool discreteClasses = false;
    std::size_t nTrain = 0, nVal = 0, nTest = 0;
    // Target-column provenance (auto-detection or explicit choice).
    int targetColUsed = 0;
    std::size_t targetDistinct = 0;
    double targetMin = 0.0, targetMax = 0.0;
    std::string targetNote;      // e.g. "auto: header name 'label'"
};

struct ExperimentResult {
    TrainingHistory history;
    NeuralNetwork net;
    bool hasNet = false;
    double trainLoss = 0.0, trainAcc = 0.0;
    double valLoss = 0.0, valAcc = 0.0;
    double testLoss = 0.0, testAcc = 0.0;
    bool hasVal = false, hasTest = false, evaluated = false;
    std::vector<std::vector<std::size_t>> confusion;
    std::size_t numClasses = 0;
    bool hasConfusion = false;
    bool confusionOnTrain = false; // true when no test set exists (tiny tables)
    double seconds = 0.0;
    bool stopped = false;
    std::string error, warning;
    PreparedData data;
};

class ExperimentController {
public:
    // Load + encode + split + normalize. Throws std::exception with a
    // UI-ready message on any dataset problem.
    static PreparedData prepare(const ExperimentConfig& cfg);
    // Check network/training settings against prepared data. Returns false
    // with err set on hard errors; warn carries non-blocking advisories
    // (loss/output mismatch hints, effective-batch notes).
    static bool validate(const ExperimentConfig& cfg, const PreparedData& data,
                         std::string& err, std::string& warn);
    // Full pipeline on the calling thread: prepare, validate, build, train
    // (streaming epochs to cb, polling stop), evaluate the held-out test set.
    // Never throws: failures land in result.error.
    static ExperimentResult run(const ExperimentConfig& cfg, TrainingCallback* cb,
                                const std::atomic<bool>* stop);
};

} // namespace miniann
