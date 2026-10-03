#include "miniann/experiment.hpp"
#include "miniann/serializer.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace miniann {

namespace {

int csvColumnCount(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return 0;
    std::string line;
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        int count = 0;
        while (std::getline(ss, cell, ',')) count++;
        return count;
    }
    return 0;
}

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

// Fit normalization on the TRAIN inputs only, apply to every split.
struct NormStats {
    int mode = 0;
    Vector scale, lo; // maxabs: scale=m; minmax: lo/hi via scale=range
    Vector hi;
};

NormStats fitNorm(const Dataset& train, int mode) {
    NormStats st;
    st.mode = mode;
    if (train.size() == 0 || mode == 0) return st;
    std::size_t dim = train.input(0).size();
    if (mode == 1) {
        double m = 0.0;
        for (std::size_t i = 0; i < train.size(); ++i)
            for (double v : train.input(i)) m = std::max(m, std::abs(v));
        st.scale = Vector(dim, m == 0.0 ? 1.0 : m);
    } else {
        st.lo = train.input(0);
        st.hi = train.input(0);
        for (std::size_t i = 1; i < train.size(); ++i) {
            const Vector& x = train.input(i);
            for (std::size_t j = 0; j < dim; ++j) {
                st.lo[j] = std::min(st.lo[j], x[j]);
                st.hi[j] = std::max(st.hi[j], x[j]);
            }
        }
    }
    return st;
}

Vector applyNorm(const Vector& x, const NormStats& st) {
    if (st.mode == 0) return x;
    Vector y = x;
    if (st.mode == 1) {
        for (std::size_t j = 0; j < y.size(); ++j) y[j] /= st.scale[j];
    } else {
        for (std::size_t j = 0; j < y.size(); ++j) {
            double r = st.hi[j] - st.lo[j];
            y[j] = (r == 0.0) ? 0.0 : (y[j] - st.lo[j]) / r;
        }
    }
    return y;
}

Dataset normDataset(const Dataset& d, const NormStats& st) {
    if (st.mode == 0) return d;
    Dataset out;
    for (std::size_t i = 0; i < d.size(); ++i)
        out.add(applyNorm(d.input(i), st), d.target(i));
    return out;
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

} // namespace

PreparedData ExperimentController::prepare(const ExperimentConfig& cfg) {
    Dataset raw;
    std::string name;
    if (cfg.dataset == "and")      { raw = makeAndGate(); name = "AND"; }
    else if (cfg.dataset == "or")  { raw = makeOrGate();  name = "OR"; }
    else if (cfg.dataset == "xor") { raw = makeXorGate(); name = "XOR"; }
    else if (cfg.dataset == "iris") {
        raw = Dataset::loadCSV("data/iris_small.csv", true, 4, 1);
        name = "Iris";
    } else if (cfg.dataset == "csv") {
        if (cfg.csvPath.empty())
            throw std::runtime_error("No CSV file selected (drop a .csv file onto the window or use Browse).");
        int col = cfg.targetCol;
        if (col < 0) {
            int total = csvColumnCount(cfg.csvPath);
            col = (total > 0) ? (total - 1) : 0;
        }
        raw = Dataset::loadCSV(cfg.csvPath, cfg.header, col, 1);
        name = "CSV";
    } else {
        throw std::invalid_argument("Unknown dataset '" + cfg.dataset + "'");
    }
    raw.validate();

    // Label encoding: single-output integer targets 0..K-1 become one-hot
    // (multiclass CSV/Iris); {0,1} stays binary; anything else is regression.
    std::size_t classes = 0;
    bool discrete = false;
    if (raw.target(0).size() == 1) {
        auto vals = distinctTargets(raw);
        bool integral = true;
        for (double v : vals)
            if (std::floor(v) != v || v < 0.0 || v > 64.0) { integral = false; break; }
        if (integral && !vals.empty()) {
            int k = int(vals.back()) + 1;
            if ((std::size_t)k == vals.size() && k >= 2 && k <= 32) {
                classes = std::size_t(k);
                discrete = true;
                if (k > 2) raw = oneHot(raw, k);
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

    if (raw.size() <= 8) {
        // Tiny truth tables: a held-out split would leave a 1-sample test
        // set (pure noise). Train and evaluate on the full table instead.
        d.tiny = true;
        NormStats st = fitNorm(raw, cfg.normMode);
        d.train = normDataset(raw, st);
        d.test = d.train;
        d.hasVal = false;
    } else {
        auto parts = raw.splitTrainValTest(cfg.trainFrac, cfg.valFrac, cfg.seed);
        NormStats st = fitNorm(std::get<0>(parts), cfg.normMode);
        d.train = normDataset(std::get<0>(parts), st);
        d.val = normDataset(std::get<1>(parts), st);
        d.test = normDataset(std::get<2>(parts), st);
        d.hasVal = true;
    }
    d.nTrain = d.train.size();
    d.nVal = d.val.size();
    d.nTest = d.test.size();
    d.train.validate();
    d.test.validate();
    if (d.hasVal) d.val.validate();
    return d;
}

bool ExperimentController::validate(const ExperimentConfig& cfg, const PreparedData& d,
                                    std::string& err, std::string& warn) {
    err.clear();
    warn.clear();
    if (cfg.hidden.empty() || cfg.hidden.size() != cfg.hiddenActs.size()) {
        err = "Need at least 1 hidden layer.";
        return false;
    }
    for (std::size_t i = 0; i < cfg.hidden.size(); ++i) {
        if (cfg.hidden[i] < 1 || cfg.hidden[i] > 1024) {
            err = "Hidden layer " + std::to_string(i + 1) + ": neuron count must be 1..1024.";
            return false;
        }
        try { ActivationFactory::create(cfg.hiddenActs[i]); }
        catch (const std::exception&) {
            err = "Unknown hidden activation '" + cfg.hiddenActs[i] + "'.";
            return false;
        }
    }
    try { ActivationFactory::create(cfg.outputAct); }
    catch (const std::exception&) {
        err = "Unknown output activation '" + cfg.outputAct + "'.";
        return false;
    }
    try { LossFactory::create(cfg.loss); }
    catch (const std::exception&) {
        err = "Unknown loss '" + cfg.loss + "'.";
        return false;
    }
    try { OptimizerFactory::create(cfg.optimizer, cfg.opt); }
    catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    if (cfg.epochs < 1 || cfg.epochs > 200000) {
        err = "Epoch count must be 1..200000.";
        return false;
    }
    if (cfg.loss == "bce" && d.outDim != 1) {
        err = "BCE needs a 1-output network (use MSE/CCE).";
        return false;
    }
    // Non-blocking advisories (§7): warn, don't block experimentation.
    std::ostringstream w;
    if (cfg.loss == "mse" && d.outDim > 1)
        w << "Warning: MSE on a " << d.outDim << "-output task — CCE usually trains classifiers better.\n";
    if (cfg.loss == "cce" && cfg.outputAct == "linear")
        w << "Warning: Linear outputs are unbounded — CCE expects [0,1]; prefer Sigmoid output.\n";
    if (cfg.loss == "bce" && cfg.outputAct != "sigmoid")
        w << "Warning: BCE is designed for Sigmoid outputs.\n";
    std::size_t eff = (cfg.batchSize == 0 || cfg.batchSize > d.nTrain) ? d.nTrain : cfg.batchSize;
    if (cfg.batchSize > d.nTrain)
        w << "Note: batch " << cfg.batchSize << " > train size " << d.nTrain
          << " — effective batch " << eff << " (full-batch).\n";
    if (d.tiny)
        w << "Note: tiny dataset — training and testing on all " << d.samples << " samples.\n";
    warn = w.str();
    if (!warn.empty() && warn.back() == '\n') warn.pop_back();
    return true;
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
        res.stopped = stop && stop->load();

        auto evalSet = [&](const Dataset& d, double& lossOut, double& accOut) {
            std::vector<Vector> p, t;
            for (std::size_t i = 0; i < d.size(); ++i) {
                p.push_back(net.predict(d.input(i)));
                t.push_back(d.target(i));
            }
            double s = 0.0;
            for (std::size_t i = 0; i < d.size(); ++i) s += loss->compute(p[i], t[i]);
            lossOut = d.size() ? s / double(d.size()) : 0.0;
            accOut = Accuracy().evaluate(p, t);
        };
        evalSet(res.data.train, res.trainLoss, res.trainAcc);
        if (res.hasVal) evalSet(res.data.val, res.valLoss, res.valAcc);
        evalSet(res.data.test, res.testLoss, res.testAcc);
        res.evaluated = true;

        if (res.data.discreteClasses && res.data.classes >= 2 && res.data.classes <= 32) {
            try {
                std::vector<Vector> p, t;
                for (std::size_t i = 0; i < res.data.test.size(); ++i) {
                    p.push_back(net.predict(res.data.test.input(i)));
                    t.push_back(res.data.test.target(i));
                }
                ConfusionMatrix cm(res.data.classes);
                cm.evaluate(p, t);
                res.confusion = cm.matrix();
                res.numClasses = res.data.classes;
                res.hasConfusion = true;
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
