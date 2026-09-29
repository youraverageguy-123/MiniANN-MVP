// and_or_demo: linearly-separable gates learn reliably with a small MLP.
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

static void runGate(const std::string& name, Dataset data,
                    const std::string& modelFile, unsigned seed) {
    std::mt19937 rng(seed);
    NeuralNetwork net;
    net.addLayer(Layer(2, 2, ActivationFactory::create("sigmoid"), rng));
    net.addLayer(Layer(1, 2, ActivationFactory::create("sigmoid"), rng));
    MSELoss loss;
    SGD opt(0.5);
    ConsoleLogger logger;
    Trainer trainer(net, loss, opt, &logger);
    TrainingConfig cfg;
    cfg.epochs = 3000; cfg.batchSize = 1; cfg.seed = seed; cfg.logEvery = 1000;
    TrainingHistory h = trainer.fit(data, nullptr, cfg);
    std::cout << "=== " << name << " final_loss=" << h.trainLoss.back() << " ===\n";
    for (std::size_t i = 0; i < data.size(); ++i) {
        Vector p = net.predict(data.input(i));
        std::cout << "  [" << data.input(i)[0] << "," << data.input(i)[1] << "] -> "
                  << std::fixed << std::setprecision(4) << p[0] << "\n";
    }
    ModelSerializer::save(net, modelFile);
}

int main() {
    runGate("AND", makeAndGate(), "and.model", 42);
    runGate("OR", makeOrGate(), "or.model", 7);
    return 0;
}
