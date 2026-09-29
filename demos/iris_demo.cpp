// iris_demo: multi-class style demo via CSV ingestion + confusion matrix.
// Uses data/iris_small.csv (4 features, 1 label in {0,1,2}).
#include <iostream>
#include <random>
#include "miniann/network.hpp"
#include "miniann/dataset.hpp"
#include "miniann/loss.hpp"
#include "miniann/optimizer.hpp"
#include "miniann/trainer.hpp"
#include "miniann/metrics.hpp"
#include "miniann/logger.hpp"
#include "miniann/serializer.hpp"

using namespace miniann;

// one-hot encode integer labels for MSE training
static Dataset oneHot(const Dataset& raw, int numClasses) {
    Dataset d;
    for (std::size_t i = 0; i < raw.size(); ++i) {
        int c = int(raw.target(i)[0]);
        Vector t(std::size_t(numClasses), 0.0);
        t[std::size_t(c)] = 1.0;
        d.add(raw.input(i), t);
    }
    return d;
}

int main() {
    std::mt19937 rng(42);
    Dataset raw;
    try {
        raw = Dataset::loadCSV("data/iris_small.csv", true, 4, 1);
    } catch (const std::exception& e) {
        std::cerr << "iris_demo: " << e.what() << " (run from MiniANN_MVP dir)\n";
        return 1;
    }
    // normalize features by max-abs for stable sigmoid/tanh training
    Dataset norm;
    for (std::size_t i = 0; i < raw.size(); ++i) {
        Vector x = raw.input(i);
        for (auto& v : x) v /= 8.0;
        norm.add(x, raw.target(i));
    }
    auto [trRaw, teRaw] = norm.split(0.8, rng);
    Dataset tr = oneHot(trRaw, 3), te = oneHot(teRaw, 3);

    NeuralNetwork net;
    net.addLayer(Layer(6, 4, ActivationFactory::create("tanh"), rng, WeightInit::Xavier));
    net.addLayer(Layer(3, 6, ActivationFactory::create("sigmoid"), rng, WeightInit::Xavier));

    MSELoss loss;
    Adam opt(0.01);
    ConsoleLogger logger;
    Trainer trainer(net, loss, opt, &logger);
    TrainingConfig cfg;
    cfg.epochs = 1500; cfg.batchSize = 8; cfg.seed = 42; cfg.logEvery = 300;
    TrainingHistory h = trainer.fit(tr, &te, cfg);
    CSVLossExporter::exportHistory(h, "iris_loss.csv");

    auto collect = [&](Dataset& d, std::vector<Vector>& p, std::vector<Vector>& t) {
        for (std::size_t i = 0; i < d.size(); ++i) {
            p.push_back(net.predict(d.input(i)));
            t.push_back(d.target(i));
        }
    };
    std::vector<Vector> pTr, tTr, pTe, tTe;
    collect(tr, pTr, tTr); collect(te, pTe, tTe);
    Accuracy acc;
    ConfusionMatrix cm(3);
    std::cout << "train acc=" << acc.evaluate(pTr, tTr) << "\n";
    std::cout << "test  acc=" << acc.evaluate(pTe, tTe) << "\n";
    cm.evaluate(pTe, tTe);
    std::cout << "confusion (rows=true, cols=pred):\n";
    cm.print();
    for (std::size_t c = 0; c < 3; ++c)
        std::cout << "class " << c << " prec=" << cm.precision(c)
                  << " rec=" << cm.recall(c) << "\n";
    ModelSerializer::save(net, "iris.model");
    return 0;
}
