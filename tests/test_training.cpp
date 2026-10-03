// spec §40: end-to-end regression tests through ExperimentController.
// 1. XOR (tiny truth table) converges to ~100% test accuracy, deterministically.
// 2. The 80/10/10 split is deterministic for a fixed seed.
// 3. Invalid configs (BCE + multi-output) are rejected with a readable error.
// 4. The Iris pipeline completes and reports separate train/val/test metrics.
#include "miniann/experiment.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace miniann;

static ExperimentConfig xorCfg() {
    ExperimentConfig c;
    c.dataset = "xor";
    c.hidden = {8};
    c.hiddenActs = {"tanh"};
    c.outputAct = "sigmoid";
    c.loss = "mse";
    c.optimizer = "adam";
    c.opt.learningRate = 0.05;
    c.epochs = 1500;
    c.batchSize = 1;
    c.seed = 42;
    c.shuffle = true;
    return c;
}

static void testXorConverges() {
    ExperimentResult r1 = ExperimentController::run(xorCfg(), nullptr, nullptr);
    assert(r1.error.empty());
    assert(r1.evaluated);
    assert(!r1.hasVal); // tiny truth table: no held-out split
    std::cout << "xor: trainAcc=" << r1.trainAcc << " testAcc=" << r1.testAcc
              << " loss=" << r1.trainLoss << "\n";
    assert(r1.testAcc >= 0.99);
    assert(r1.trainAcc >= 0.99);
    // Deterministic: same seed twice -> identical outcome.
    ExperimentResult r2 = ExperimentController::run(xorCfg(), nullptr, nullptr);
    assert(std::abs(r1.testAcc - r2.testAcc) < 1e-12);
    assert(r2.history.trainLoss.size() == r1.history.trainLoss.size());
    // Confusion matrix over the 4 truth-table rows.
    assert(r1.hasConfusion && r1.numClasses == 2);
    std::size_t total = 0;
    for (auto& row : r1.confusion)
        for (auto v : row) total += v;
    assert(total == 4);
    std::cout << "PASS xor-converges\n";
}

static void testSplitDeterministic() {
    Dataset d;
    for (int i = 0; i < 100; ++i) d.add({(double)i}, {(double)(i % 2)});
    auto a = d.splitTrainValTest(0.8, 0.1, 42);
    auto b = d.splitTrainValTest(0.8, 0.1, 42);
    auto c = d.splitTrainValTest(0.8, 0.1, 7);
    assert(std::get<0>(a).size() == 80 && std::get<1>(a).size() == 10 && std::get<2>(a).size() == 10);
    for (std::size_t i = 0; i < 80; ++i)
        assert(std::get<0>(a).input(i)[0] == std::get<0>(b).input(i)[0]); // same seed: identical
    bool differs = false;
    for (std::size_t i = 0; i < 80; ++i)
        if (std::get<0>(a).input(i)[0] != std::get<0>(c).input(i)[0]) differs = true;
    assert(differs); // different seed: (almost surely) different order
    std::cout << "PASS split-deterministic\n";
}

static void testValidationRejectsBceMulti() {
    ExperimentConfig c = xorCfg();
    c.dataset = "iris";
    c.hidden = {6};
    c.hiddenActs = {"tanh"};
    c.loss = "bce"; // 3-output data + BCE must fail with a readable error
    ExperimentResult r = ExperimentController::run(c, nullptr, nullptr);
    assert(!r.error.empty());
    assert(r.error.find("BCE") != std::string::npos);
    std::cout << "PASS validation-rejects-bce-multi (" << r.error << ")\n";
}

static void testIrisPipeline() {
    ExperimentConfig c;
    c.dataset = "iris";
    c.hidden = {8, 8};
    c.hiddenActs = {"relu", "relu"};
    c.outputAct = "sigmoid";
    c.loss = "cce";
    c.optimizer = "adam";
    c.opt.learningRate = 0.01;
    c.epochs = 300;
    c.batchSize = 8;
    c.seed = 42;
    ExperimentResult r = ExperimentController::run(c, nullptr, nullptr);
    assert(r.error.empty());
    assert(r.evaluated && r.hasVal);
    assert(r.data.nTrain + r.data.nVal + r.data.nTest == r.data.samples);
    assert(r.hasConfusion && r.numClasses == 3);
    assert(r.trainAcc >= 0.0 && r.trainAcc <= 1.0);
    assert(r.testAcc >= 0.0 && r.testAcc <= 1.0);
    std::cout << "iris: train=" << r.trainAcc << " val=" << r.valAcc << " test=" << r.testAcc << "\n";
    std::cout << "PASS iris-pipeline\n";
}

static void testAndOrGates() {
    for (const char* gate : {"and", "or"}) {
        ExperimentConfig c = xorCfg();
        c.dataset = gate;
        c.loss = "bce";
        c.optimizer = "sgd";
        c.opt.learningRate = 0.05;
        c.epochs = 1000;
        ExperimentResult r = ExperimentController::run(c, nullptr, nullptr);
        assert(r.error.empty());
        std::cout << gate << ": trainAcc=" << r.trainAcc << " testAcc=" << r.testAcc << "\n";
        assert(r.trainAcc >= 0.99 && r.testAcc >= 0.99);
    }
    std::cout << "PASS and-or-gates\n";
}

static void testOptimizersDescend() {
    // Every optimizer must move weights and reduce loss on a trivial task.
    for (const char* opt : {"sgd", "momentum", "adam"}) {
        Dataset d = makeXorGate();
        std::mt19937 rng(1);
        NeuralNetwork net;
        net.addLayer(Layer(4, 2, ActivationFactory::create("tanh"), rng, WeightInit::Xavier));
        net.addLayer(Layer(1, 4, ActivationFactory::create("sigmoid"), rng, WeightInit::Xavier));
        auto loss = LossFactory::create("mse");
        auto optimizer = OptimizerFactory::create(opt, 0.05);
        auto totalLoss = [&]() {
            double s = 0.0;
            for (std::size_t i = 0; i < d.size(); ++i)
                s += loss->compute(net.predict(d.input(i)), d.target(i));
            return s / d.size();
        };
        double before = totalLoss();
        double w0 = net.layers()[0].neurons()[0].weights()[0];
        Trainer trainer(net, *loss, *optimizer, nullptr);
        TrainingConfig cfg;
        cfg.epochs = 25;
        cfg.batchSize = 4;
        cfg.shuffle = false;
        trainer.fit(d, nullptr, cfg, nullptr);
        double after = totalLoss();
        double w1 = net.layers()[0].neurons()[0].weights()[0];
        std::cout << opt << ": loss " << before << " -> " << after << "\n";
        assert(w0 != w1);   // weights moved
        assert(after < before); // loss decreased
    }
    std::cout << "PASS optimizers-descend\n";
}

int main() {
    testXorConverges();
    testAndOrGates();
    testOptimizersDescend();
    testSplitDeterministic();
    testValidationRejectsBceMulti();
    testIrisPipeline();
    std::cout << "ALL TRAINING-PIPE CHECKS PASS\n";
    return 0;
}
