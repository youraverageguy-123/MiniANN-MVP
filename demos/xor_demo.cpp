// xor_demo: proves non-linear learning (2-4-1 tanh+sigmoid, SGD).
#include <iostream>
#include <iomanip>
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

int main() {
    std::mt19937 rng(42);
    Dataset data = makeXorGate();
    data.validate();

    NeuralNetwork net;
    net.addLayer(Layer(4, 2, ActivationFactory::create("tanh"), rng, WeightInit::Xavier));
    net.addLayer(Layer(1, 4, ActivationFactory::create("sigmoid"), rng, WeightInit::Xavier));

    MSELoss loss;
    SGD opt(0.5);
    ConsoleLogger logger;
    Trainer trainer(net, loss, opt, &logger);

    TrainingConfig cfg;
    cfg.epochs = 10000;
    cfg.batchSize = 1;
    cfg.shuffle = true;
    cfg.seed = 42;
    cfg.logEvery = 2000;

    TrainingHistory hist = trainer.fit(data, nullptr, cfg);
    CSVLossExporter::exportHistory(hist, "xor_loss.csv");

    // Evaluate
    std::vector<Vector> preds, targets;
    std::cout << "XOR results (2-4-1 tanh/sigmoid, SGD lr=0.5):\n";
    for (std::size_t i = 0; i < data.size(); ++i) {
        Vector p = net.predict(data.input(i));
        preds.push_back(p);
        targets.push_back(data.target(i));
        std::cout << "  [" << data.input(i)[0] << "," << data.input(i)[1] << "] -> "
                  << std::fixed << std::setprecision(4) << p[0]
                  << " (target " << data.target(i)[0] << ")\n";
    }
    Accuracy acc;
    std::cout << "accuracy=" << acc.evaluate(preds, targets) * 100.0 << "% "
              << "final_loss=" << hist.trainLoss.back() << "\n";

    ModelSerializer::save(net, "xor.model");
    NeuralNetwork net2 = ModelSerializer::load("xor.model", rng);
    double maxDiff = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        Vector a = net.predict(data.input(i));
        Vector b = net2.predict(data.input(i));
        maxDiff = std::max(maxDiff, std::abs(a[0] - b[0]));
    }
    std::cout << "serialization round-trip max|y_orig - y_loaded|=" << maxDiff << "\n";
    return 0;
}
