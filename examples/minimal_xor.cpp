// Minimal library usage: the whole public API in ~30 lines.
// Compile against the installed library:
//   g++ -std=c++17 minimal_xor.cpp -lminiann -o minimal_xor
// or in-repo: g++ -std=c++17 -Iinclude examples/minimal_xor.cpp <src/*.cpp>
#include <iostream>
#include <random>
#include "miniann/miniann.hpp"

using namespace miniann;

int main() {
    std::mt19937 rng(42);
    Dataset data = makeXorGate();

    NeuralNetwork net; // Network owns Layers, Layers own Neurons (composition)
    net.addLayer(Layer(4, 2, ActivationFactory::create("tanh"), rng, WeightInit::Xavier, 1));
    net.addLayer(Layer(1, 4, ActivationFactory::create("sigmoid"), rng));

    MSELoss loss;           // swappable Strategy behind ILoss
    SGD opt(0.5);           // swappable Strategy behind IOptimizer
    Trainer trainer(net, loss, opt, nullptr);

    TrainingConfig cfg;
    cfg.epochs = 5000;
    cfg.batchSize = 1;
    cfg.seed = 42;
    cfg.logEvery = 100000;
    trainer.fit(data, nullptr, cfg, nullptr);

    Accuracy acc; // threshold ctor arg exists for tuning; default 0.5
    std::vector<Vector> p, t;
    for (std::size_t i = 0; i < data.size(); ++i) {
        p.push_back(net.predict(data.input(i)));
        t.push_back(data.target(i));
    }
    std::cout << "xor acc=" << acc.evaluate(p, t) * 100.0 << "%\n";
    ModelSerializer::save(net, "minimal.model"); // MINIANN 2, v1-readable
    return 0;
}
