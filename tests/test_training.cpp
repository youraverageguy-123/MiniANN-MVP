// spec §40: end-to-end regression tests through ExperimentController.
// Tiny truth tables report TRAIN metrics only (no held-out test exists).
#include "miniann/experiment.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
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
    assert(!r1.hasVal);
    assert(!r1.hasTest); // tiny truth table: NO test set exists, none reported
    std::cout << "xor: trainAcc=" << r1.trainAcc << " loss=" << r1.trainLoss << "\n";
    assert(r1.trainAcc >= 0.99);
    // Deterministic: same seed twice -> identical outcome.
    ExperimentResult r2 = ExperimentController::run(xorCfg(), nullptr, nullptr);
    assert(std::abs(r1.trainAcc - r2.trainAcc) < 1e-12);
    assert(r2.history.trainLoss.size() == r1.history.trainLoss.size());
    // Confusion matrix over the 4 truth-table rows, honestly labeled train-set.
    assert(r1.hasConfusion && r1.numClasses == 2 && r1.confusionOnTrain);
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
        std::cout << gate << ": trainAcc=" << r.trainAcc << " hasTest=" << r.hasTest << "\n";
        assert(r.trainAcc >= 0.99 && !r.hasTest); // tiny: train-only, no test metric
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

static void testNineSampleSplit() {
    // 9 samples broke fixed-fraction splitters (0-sample val part).
    // Min-1 counts must yield a usable 7/1/1 split instead of an error.
    Dataset d;
    for (int i = 0; i < 9; ++i) d.add({(double)i, (double)(i % 2)}, {(double)(i % 2)});
    auto parts = d.splitCounts(7, 1, 42, true);
    assert(std::get<0>(parts).size() == 7);
    assert(std::get<1>(parts).size() == 1);
    assert(std::get<2>(parts).size() == 1);
    // Full controller path via a throwaway CSV: 9 rows must train, not error.
    {
        std::ofstream f("tmp_edge9.csv");
        f << "x1,x2,y\n";
        for (int i = 0; i < 9; ++i) f << (i % 3) << "," << (i % 2) << "," << (i % 2) << "\n";
    }
    ExperimentConfig c = xorCfg();
    c.dataset = "csv";
    c.csvPath = "tmp_edge9.csv";
    c.epochs = 50;
    ExperimentResult r = ExperimentController::run(c, nullptr, nullptr);
    std::remove("tmp_edge9.csv");
    assert(r.error.empty());
    assert(r.hasTest && r.evaluated);
    assert(r.data.nTrain + r.data.nVal + r.data.nTest == 9);
    assert(r.data.nVal >= 1 && r.data.nTest >= 1 && r.data.nTrain >= 1);
    std::cout << "PASS nine-sample-split (" << r.data.nTrain << "/" << r.data.nVal
              << "/" << r.data.nTest << ")\n";
}

static void testCustomSplitCounts() {
    Dataset d;
    for (int i = 0; i < 100; ++i) d.add({(double)i}, {(double)(i % 2)});
    auto parts = d.splitCounts(70, 20, 42, true);
    assert(std::get<0>(parts).size() == 70);
    assert(std::get<1>(parts).size() == 20);
    assert(std::get<2>(parts).size() == 10);
    // Sequential (unshuffled) split: first rows train, reproducible, in order.
    auto s1 = d.splitCounts(70, 20, 42, false);
    auto s2 = d.splitCounts(70, 20, 99, false);
    for (std::size_t i = 0; i < 70; ++i) {
        assert(std::get<0>(s1).input(i)[0] == (double)i);
        assert(std::get<0>(s1).input(i)[0] == std::get<0>(s2).input(i)[0]);
    }
    std::cout << "PASS custom-split-counts\n";
}

static void testManyClassWarning() {
    // 40 distinct integer targets: ambiguous (40-class problem vs integer
    // regression). Must NOT error and must NOT silently one-hot: proceed as
    // regression with an explanatory note.
    {
        std::ofstream f("tmp_edge40.csv");
        f << "x,y\n";
        for (int i = 0; i < 40; ++i) f << (i * 0.5) << "," << i << "\n";
    }
    ExperimentConfig c = xorCfg();
    c.dataset = "csv";
    c.csvPath = "tmp_edge40.csv";
    PreparedData p = ExperimentController::prepare(c);
    std::remove("tmp_edge40.csv");
    assert(!p.discreteClasses);
    assert(p.splitNote.find("regression") != std::string::npos);
    std::cout << "PASS many-class-warning\n";
}

static void testConstantTargetRejected() {
    // A constant target column is trivially 100% predictable: refuse loudly
    // instead of reporting vacuous metrics. (Header "y" auto-matches, so the
    // constant column is actually selected here.)
    {
        std::ofstream f("tmp_edge_const.csv");
        f << "x,y\n";
        for (int i = 0; i < 40; ++i) f << i << ",5\n";
    }
    ExperimentConfig c = xorCfg();
    c.dataset = "csv";
    c.csvPath = "tmp_edge_const.csv";
    bool threw = false;
    std::string msg;
    try {
        PreparedData p = ExperimentController::prepare(c);
        (void)p;
    } catch (const std::exception& e) {
        threw = true;
        msg = e.what();
    }
    std::remove("tmp_edge_const.csv");
    assert(threw);
    assert(msg.find("constant") != std::string::npos);
    std::cout << "PASS constant-target-rejected\n";
}

static void testAutoPicksLabelColumn() {
    // No helpful header names: the first 2..32-class integral column wins.
    {
        std::ofstream f("tmp_edge_auto.csv");
        f << "a,b,c\n";
        for (int i = 0; i < 40; ++i) f << (i % 3) << "," << i << "," << (i % 2) << "\n";
    }
    ExperimentConfig c = xorCfg();
    c.dataset = "csv";
    c.csvPath = "tmp_edge_auto.csv";
    PreparedData p = ExperimentController::prepare(c);
    assert(p.targetColUsed == 0);
    assert(p.outDim == 3); // one-hot encoded 3 classes
    assert(p.targetNote.find("3-class") != std::string::npos);
    // Explicit choice wins over auto-detection.
    c.targetCol = 2;
    PreparedData p2 = ExperimentController::prepare(c);
    std::remove("tmp_edge_auto.csv");
    assert(p2.targetColUsed == 2);
    assert(p2.outDim == 1);
    assert(p2.discreteClasses && p2.classes == 2);
    std::cout << "PASS auto-picks-label-column\n";
}

int main() {
    testXorConverges();
    testAndOrGates();
    testNineSampleSplit();
    testCustomSplitCounts();
    testManyClassWarning();
    testConstantTargetRejected();
    testAutoPicksLabelColumn();
    testOptimizersDescend();
    testSplitDeterministic();
    testValidationRejectsBceMulti();
    testIrisPipeline();
    std::cout << "ALL TRAINING-PIPE CHECKS PASS\n";
    return 0;
}
