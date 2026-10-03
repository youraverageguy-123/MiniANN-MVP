#include "miniann/experiment.hpp"
#include "miniann/datasource.hpp"
#include "miniann/normalize.hpp"
#include "miniann/serializer.hpp"
#include "miniann/validate.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>

namespace miniann {

// Distinct target values of a single-output dataset.
std::vector<double> distinctTargets(const Dataset& d) {
    std::vector<double> vals;
    for (std::size_t i = 0; i < d.size(); ++i) vals.push_back(d.target(i)[0]);
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
    return vals;
}

Dataset oneHot(const Dataset& raw, int k) {
    Dataset d;
    for (std::size_t i = 0; i < raw.size(); ++i) {
        int c = int(raw.target(i)[0]);
        Vector t(std::size_t(k), 0.0);
        if (c >= 0 && c < k) t[std::size_t(c)] = 1.0;
        d.add(raw.input(i), t);
    }
    return d;
}

// Forwards epochs to the UI callback and exposes the stop flag to the trainer.
struct ControlCallback : public TrainingCallback {
    TrainingCallback* ui = nullptr;
    const std::atomic<bool>* stop = nullptr;
    ControlCallback(TrainingCallback* u, const std::atomic<bool>* s) : ui(u), stop(s) {}
    void onEpoch(int epoch, const TrainingHistory& hist) override {
        if (ui) ui->onEpoch(epoch, hist);
    }
    bool shouldStop() const override {
        return stop && stop->load();
    }
};

PreparedData ExperimentController::prepare(const ExperimentConfig& cfg) {
    Dataset raw;
    std::string name;
    int srcCol = 0;
    std::string srcNote;
    try {
        // Dataset kinds are polymorphic sources behind a factory: adding one
        // never touches this function (Open/Closed Principle in action).
        auto source = DatasetSourceFactory::create(cfg.dataset);
        LoadedRaw loaded = source->load(cfg);
        raw = std::move(loaded.data);
        name = loaded.name;
        srcCol = loaded.targetCol;
        srcNote = loaded.note;
        raw.validate();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Dataset '") + (name.empty() ? cfg.dataset : name) +
                                 "' failed to load: " + e.what());
    }

    // Target stats from the SOURCE values (pre-encoding) so the UI can show
    // exactly what is being predicted (column, distinct values, range).
    int tgtDistinct = 0;
    double tgtMin = 0.0, tgtMax = 0.0;
    if (!raw.target(0).empty()) {
        auto tvals = distinctTargets(raw);
        tgtDistinct = (int)tvals.size();
        tgtMin = tvals.front();
        tgtMax = tvals.back();
    }
    // Label encoding: single-output integer targets 0..K-1 become one-hot
    // (multiclass CSV/Iris); {0,1} stays binary; anything else is regression.
    // More than 32 distinct integer targets is ambiguous (classification with
    // many classes vs. integer regression): proceed as regression with a note
    // rather than guessing wrong silently.
    std::size_t classes = 0;
    bool discrete = false;
    std::string encodeNote;
    if (raw.target(0).size() == 1) {
        auto vals = distinctTargets(raw);
        bool integral = true;
        for (double v : vals)
            if (std::floor(v) != v || v < 0.0 || v > 1000000.0) { integral = false; break; }
        if (integral && !vals.empty()) {
            int k = int(vals.back()) + 1;
            if ((std::size_t)k == vals.size() && k >= 2 && k <= 32) {
                classes = std::size_t(k);
                discrete = true;
                if (k > 2) raw = oneHot(raw, k);
            } else if ((std::size_t)k == vals.size() && k > 32) {
                std::ostringstream ns;
                ns << "Note: target has " << k << " distinct integer values — treating as regression. "
                   << "One-hot encode the labels for classification with many classes.";
                encodeNote = ns.str();
            }
        }
    } else {
        classes = raw.target(0).size();
        discrete = true; // already one-hot encoded
    }

    PreparedData d;
    d.name = name;
    d.samples = raw.size();
    d.inDim = raw.input(0).size();
    d.outDim = raw.target(0).size();
    d.features = d.inDim;
    d.classes = classes;
    d.discreteClasses = discrete;
    d.splitNote = encodeNote;
    d.targetColUsed = srcCol;
    d.targetNote = srcNote;
    d.targetDistinct = (std::size_t)std::max(0, tgtDistinct);
    d.targetMin = tgtMin;
    d.targetMax = tgtMax;

    if (raw.size() <= 8) {
        // Tiny truth tables: no held-out split exists (any split would leave
        // 0- or 1-sample parts). Train on the full table; there is NO test
        // set, so no test metric will be reported — only train metrics.
        d.tiny = true;
        auto normalizer = NormalizerFactory::create(cfg.normMode);
        normalizer->fit(raw);
        d.train = normalizer->normalize(raw);
        d.hasVal = false;
        d.hasTest = false;
        d.splitNote += (d.splitNote.empty() ? std::string() : std::string("\n")) +
            "Tiny dataset: training on all " + std::to_string(d.samples) +
            " samples; no held-out test set, so no test metric is reported.";
    } else {
        // Exact counts with a minimum of 1 sample per part (a zero val share
        // disables validation), so sizes like 9 samples (7/1/1) or extreme
        // ratios can never produce an empty split. Requested shares are
        // honored whenever they fit.
        std::size_t reqVal = (cfg.valFrac <= 0.0) ? 0
            : std::max<std::size_t>(1, std::size_t(double(raw.size()) * cfg.valFrac));
        std::size_t wantTrain = std::size_t(double(raw.size()) * cfg.trainFrac);
        std::size_t nVal = reqVal;
        std::size_t nTest = (wantTrain + nVal < raw.size()) ? raw.size() - wantTrain - nVal : 1;
        std::size_t nTrain = raw.size() - nVal - nTest;
        if (nTrain < 1)
            throw std::runtime_error("Dataset '" + name + "' is too small for a train/val/test split.");
        if (nTrain != wantTrain || nVal != reqVal) {
            std::ostringstream ns;
            ns << "Note: split adjusted to " << nTrain << "/" << nVal << "/" << nTest
               << " to guarantee non-empty train" << (reqVal > 0 ? "/val" : "") << "/test parts.";
            d.splitNote += (d.splitNote.empty() ? std::string() : std::string("\n")) + ns.str();
        }
        auto parts = raw.splitCounts(nTrain, nVal, cfg.seed, cfg.splitShuffle);
        auto normalizer = NormalizerFactory::create(cfg.normMode);
        normalizer->fit(std::get<0>(parts));
        d.train = normalizer->normalize(std::get<0>(parts));
        d.val = normalizer->normalize(std::get<1>(parts));
        d.test = normalizer->normalize(std::get<2>(parts));
        d.hasVal = (nVal > 0);
        d.hasTest = true;
    }
    d.nTrain = d.train.size();
    d.nVal = d.val.size();
    d.nTest = d.test.size();
    try {
        d.train.validate();
        if (d.hasVal) d.val.validate();
        if (d.hasTest) d.test.validate();
    } catch (const std::exception& e) {
        throw std::runtime_error("Dataset '" + name + "' failed validation: " + e.what());
    }
    return d;
}

bool ExperimentController::validate(const ExperimentConfig& cfg, const PreparedData& d,
                                    std::string& err, std::string& warn) {
    // Delegated to the rule composite: one independently-testable object per
    // check instead of a monolithic if-chain.
    return ConfigValidator::defaults().validate(cfg, d, err, warn);
}

ExperimentResult ExperimentController::run(const ExperimentConfig& cfg, TrainingCallback* cb,
                                           const std::atomic<bool>* stop) {
    ExperimentResult res;
    auto t0 = std::chrono::steady_clock::now();
    try {
        res.data = prepare(cfg);
        if (!validate(cfg, res.data, res.error, res.warning)) return res;

        std::mt19937 rng(cfg.seed);
        NeuralNetwork net;
        std::size_t prev = res.data.inDim;
        for (std::size_t l = 0; l < cfg.hidden.size(); ++l) {
            net.addLayer(Layer(std::size_t(cfg.hidden[l]), prev,
                               ActivationFactory::create(cfg.hiddenActs[l]),
                               rng, WeightInit::Xavier));
            prev = std::size_t(cfg.hidden[l]);
        }
        net.addLayer(Layer(res.data.outDim, prev,
                           ActivationFactory::create(cfg.outputAct), rng, WeightInit::Xavier));

        auto loss = LossFactory::create(cfg.loss);
        auto opt = OptimizerFactory::create(cfg.optimizer, cfg.opt);

        TrainingConfig tc;
        tc.epochs = cfg.epochs;
        tc.batchSize = cfg.batchSize;
        tc.shuffle = cfg.shuffle;
        tc.seed = cfg.seed;
        tc.logEvery = 1;

        ControlCallback ctl{cb, stop};
        Trainer trainer(net, *loss, *opt, nullptr);
        res.history = trainer.fit(res.data.train,
                                  res.data.hasVal ? &res.data.val : nullptr, tc, &ctl);
        res.net = net;
        res.hasNet = true;
        res.hasVal = res.data.hasVal;
        res.hasTest = res.data.hasTest;
        res.stopped = stop && stop->load();
        if (!res.data.splitNote.empty())
            res.warning += (res.warning.empty() ? "" : "\n") + res.data.splitNote;

        auto evalSet = [&](const Dataset& d, double& lossOut, double& accOut) {
            std::vector<Vector> p, t;
            for (std::size_t i = 0; i < d.size(); ++i) {
                p.push_back(net.predict(d.input(i)));
                t.push_back(d.target(i));
            }
            double s = 0.0;
            for (std::size_t i = 0; i < d.size(); ++i) s += loss->compute(p[i], t[i]);
            lossOut = d.size() ? s / double(d.size()) : 0.0;
            accOut = p.empty() ? 0.0 : Accuracy().evaluate(p, t);
        };
        evalSet(res.data.train, res.trainLoss, res.trainAcc);
        if (res.hasVal) evalSet(res.data.val, res.valLoss, res.valAcc);
        // The test set is evaluated ONCE, after training, and only when a
        // genuine held-out set exists. Tiny truth tables have none: no test
        // metric is reported for them, ever.
        if (res.hasTest) evalSet(res.data.test, res.testLoss, res.testAcc);
        res.evaluated = true;

        if (res.data.discreteClasses && res.data.classes >= 2 && res.data.classes <= 32) {
            try {
                const Dataset& cd = res.hasTest ? res.data.test : res.data.train;
                std::vector<Vector> p, t;
                for (std::size_t i = 0; i < cd.size(); ++i) {
                    p.push_back(net.predict(cd.input(i)));
                    t.push_back(cd.target(i));
                }
                ConfusionMatrix cm(res.data.classes);
                cm.evaluate(p, t);
                res.confusion = cm.matrix();
                res.numClasses = res.data.classes;
                res.hasConfusion = true;
                res.confusionOnTrain = !res.hasTest;
            } catch (const std::exception&) { /* confusion stays unavailable */ }
        }
    } catch (const std::exception& e) {
        res.error = e.what();
    }
    auto t1 = std::chrono::steady_clock::now();
    res.seconds = std::chrono::duration<double>(t1 - t0).count();
    return res;
}

} // namespace miniann
