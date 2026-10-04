// Design-pattern regression tests: every Strategy / Factory / Composite /
// Chain / Visitor introduced for the OOP overhaul is exercised directly.
#include "miniann/activation.hpp"
#include "miniann/dataset.hpp"
#include "miniann/datasource.hpp"
#include "miniann/experiment.hpp"
#include "miniann/loss.hpp"
#include "miniann/network.hpp"
#include "miniann/normalize.hpp"
#include "miniann/optimizer.hpp"
#include "miniann/validate.hpp"
#include "miniann/scheduler.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace miniann;

static Dataset toyData() {
    Dataset d;
    d.add({0, 0}, {0});
    d.add({0, 10}, {0});
    d.add({10, 0}, {1});
    d.add({10, 10}, {1});
    return d;
}

static void testNormalizers() {
    Dataset d = toyData();
    auto none = NormalizerFactory::create(0);
    none->fit(d);
    assert(none->apply({4, 6}) == Vector({4, 6}));
    assert(none->name() == "none");

    auto mx = NormalizerFactory::create("maxabs");
    mx->fit(d);
    Vector a = mx->apply({10, 5});
    assert(std::abs(a[0] - 1.0) < 1e-12 && std::abs(a[1] - 0.5) < 1e-12);
    Dataset dn = mx->normalize(d);
    assert(std::abs(dn.input(2)[0] - 1.0) < 1e-12);

    auto mm = NormalizerFactory::create(2);
    mm->fit(d);
    Vector b = mm->apply({10, 5});
    assert(std::abs(b[0] - 1.0) < 1e-12 && std::abs(b[1] - 0.5) < 1e-12);
    // Constant feature maps to 0 instead of dividing by zero.
    Dataset c;
    c.add({3, 3}, {0});
    c.add({3, 7}, {1});
    mm->fit(c);
    assert(mm->apply({3, 5})[0] == 0.0);

    bool threw = false;
    try { NormalizerFactory::create(42); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    threw = false;
    try { NormalizerFactory::create("zscore"); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    std::cout << "PASS normalizer-strategy\n";
}

static void testWeightInits() {
    for (const char* n : {"uniform", "xavier", "he"}) {
        auto init = WeightInitFactory::create(n);
        std::mt19937 rng(7);
        Vector w(4, 0.0);
        double b = 1234.0;
        init->initialize(w, b, 4, 2, rng);
        for (double v : w) assert(std::isfinite(v));
        assert(std::isfinite(b));
        assert(init->name() == n);
    }
    auto byEnum = WeightInitFactory::create(WeightInit::He);
    assert(byEnum->name() == "he");
    bool threw = false;
    try { WeightInitFactory::create("glorot"); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    // End to end: every scheme builds a working neuron.
    for (WeightInit m : {WeightInit::Uniform, WeightInit::Xavier, WeightInit::He}) {
        std::mt19937 rng(7);
        Neuron n(2, 1, ActivationFactory::create("tanh"), rng, m);
        assert(std::isfinite(n.forward({0.5, -0.5})));
    }
    std::cout << "PASS weight-init-strategy\n";
}

static ExperimentConfig baseCfg() {
    ExperimentConfig c;
    c.dataset = "xor";
    c.hidden = {8};
    c.hiddenActs = {"tanh"};
    return c;
}

static PreparedData dummyData() {
    PreparedData d;
    d.outDim = 1;
    d.nTrain = 4;
    d.samples = 4;
    return d;
}

static void testValidatorRules() {
    // Each hard-error rule fires on its own trigger...
    {
        ExperimentConfig c = baseCfg();
        c.hidden.clear();
        PreparedData d = dummyData();
        std::string err, warn;
        assert(!ExperimentController::validate(c, d, err, warn));
        assert(err.find("hidden layer") != std::string::npos);
    }
    {
        ExperimentConfig c = baseCfg();
        c.hidden = {0};
        PreparedData d = dummyData();
        std::string err, warn;
        assert(!ExperimentController::validate(c, d, err, warn));
        assert(err.find("neuron count") != std::string::npos);
    }
    {
        ExperimentConfig c = baseCfg();
        c.hiddenActs = {"swish2"};
        PreparedData d = dummyData();
        std::string err, warn;
        assert(!ExperimentController::validate(c, d, err, warn));
        assert(err.find("activation") != std::string::npos);
    }
    {
        ExperimentConfig c = baseCfg();
        c.loss = "huber";
        PreparedData d = dummyData();
        std::string err, warn;
        assert(!ExperimentController::validate(c, d, err, warn));
        assert(err.find("loss") != std::string::npos);
    }
    {
        ExperimentConfig c = baseCfg();
        c.epochs = 0;
        PreparedData d = dummyData();
        std::string err, warn;
        assert(!ExperimentController::validate(c, d, err, warn));
        assert(err.find("Epoch") != std::string::npos);
    }
    {
        ExperimentConfig c = baseCfg();
        c.loss = "bce";
        PreparedData d = dummyData();
        d.outDim = 3;
        std::string err, warn;
        assert(!ExperimentController::validate(c, d, err, warn));
        assert(err.find("BCE") != std::string::npos);
    }
    // ...advisories warn without blocking...
    {
        ExperimentConfig c = baseCfg();
        c.loss = "mse";
        PreparedData d = dummyData();
        d.outDim = 3;
        std::string err, warn;
        assert(ExperimentController::validate(c, d, err, warn));
        assert(warn.find("MSE") != std::string::npos);
    }
    // ...and a valid config passes silently.
    {
        ExperimentConfig c = baseCfg();
        PreparedData d = dummyData();
        std::string err, warn;
        assert(ExperimentController::validate(c, d, err, warn));
        assert(err.empty() && warn.empty());
        assert(ConfigValidator::defaults().ruleCount() >= 12);
    }
    // Single rules are directly unit-testable too.
    {
        BceOutputRule rule;
        ValidationReport rep;
        ExperimentConfig c = baseCfg();
        c.loss = "bce";
        PreparedData d = dummyData();
        d.outDim = 2;
        rule.check(c, d, rep);
        assert(rep.errors.size() == 1 && rep.warnings.empty());
        assert(rule.name() == "bce-output");
    }
    std::cout << "PASS validator-composite\n";
}

static void testDatasetSources() {
    for (const char* k : {"and", "or", "xor", "iris"}) {
        auto src = DatasetSourceFactory::create(k);
        assert(src->key() == k);
        ExperimentConfig c;
        c.dataset = k;
        LoadedRaw lr = src->load(c);
        assert(lr.data.size() > 0);
        assert(!lr.name.empty());
    }
    auto csv = DatasetSourceFactory::create("csv");
    assert(csv->key() == "csv");
    bool threw = false;
    try { DatasetSourceFactory::create("mnist"); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    threw = false;
    try {
        ExperimentConfig c;
        c.dataset = "csv";
        csv->load(c);
    } catch (const std::exception&) { threw = true; }
    assert(threw); // empty path rejected
    std::cout << "PASS dataset-source-factory\n";
}

static void testTargetSelectorChain() {
    TargetScan scan;
    scan.cols = 3;
    scan.headerNames = {"age", "label", "score"};
    for (int i = 0; i < 3; ++i) scan.columns.push_back(ColumnProfile());
    scan.columns[0].distinct = {"20", "30", "40"}; // label-like but later header wins
    scan.columns[1].distinct = {"0", "1"};
    scan.columns[2].capped = true;
    TargetSelectorChain chain = TargetSelectorChain::defaults();
    assert(chain.linkCount() == 3);
    TargetChoice pick = chain.select(scan);
    assert(pick.column == 1); // header name beats the earlier distinct column
    // Without helpful headers the distinct-value link decides.
    TargetScan scan2;
    scan2.cols = 2;
    scan2.columns.push_back(ColumnProfile());
    scan2.columns.push_back(ColumnProfile());
    scan2.columns[1].distinct = {"0", "1", "2"};
    TargetChoice pick2 = chain.select(scan2);
    assert(pick2.column == 1);
    // Nothing label-like: last-column fallback always claims.
    TargetScan scan3;
    scan3.cols = 2;
    scan3.columns.push_back(ColumnProfile());
    scan3.columns.push_back(ColumnProfile());
    scan3.columns[0].capped = true;
    scan3.columns[1].capped = true;
    TargetChoice pick3 = chain.select(scan3);
    assert(pick3.column == 1);
    // Empty scan: nobody claims.
    TargetScan empty;
    assert(chain.select(empty).column == -1);
    std::cout << "PASS target-selector-chain\n";
}

static void testSplitOrders() {
    Dataset d;
    for (int i = 0; i < 10; ++i) d.add({(double)i}, {(double)(i % 2)});
    Dataset::ShuffledOrder sh;
    Dataset::SequentialOrder sq;
    auto a = d.splitCounts(7, 2, sh, 42);
    auto b = d.splitCounts(7, 2, sh, 42);
    for (std::size_t i = 0; i < 7; ++i)
        assert(std::get<0>(a).input(i)[0] == std::get<0>(b).input(i)[0]); // seeded: identical
    auto s = d.splitCounts(7, 2, sq, 42);
    for (std::size_t i = 0; i < 7; ++i)
        assert(std::get<0>(s).input(i)[0] == (double)i); // sequential: identity order
    std::cout << "PASS split-order-strategy\n";
}

struct CountingVisitor : public INeuronVisitor {
    int count = 0;
    std::vector<std::pair<std::size_t, std::size_t>> order;
    void visit(std::size_t l, std::size_t j, Neuron&) override {
        count++;
        order.emplace_back(l, j);
    }
};

static void testVisitorTraversal() {
    std::mt19937 rng(3);
    NeuralNetwork net;
    net.addLayer(Layer(3, 2, ActivationFactory::create("relu"), rng, WeightInit::Xavier));
    net.addLayer(Layer(1, 3, ActivationFactory::create("sigmoid"), rng, WeightInit::Xavier));
    CountingVisitor v;
    net.accept(v);
    assert(v.count == 4); // every neuron exactly once
    assert((v.order[0] == std::pair<std::size_t, std::size_t>(0, 0)));
    assert((v.order[3] == std::pair<std::size_t, std::size_t>(1, 0))); // layer order preserved
    std::cout << "PASS visitor-traversal\n";
}

static void testLRSchedulers() {
    auto constSched = SchedulerFactory::create("constant");
    assert(constSched->name() == "Constant");
    assert(std::abs(constSched->getRate(0.1, 50, 100) - 0.1) < 1e-9);

    auto stepSched = SchedulerFactory::create("step", 100, 0.5);
    assert(stepSched->name() == "Step Decay");
    assert(std::abs(stepSched->getRate(0.1, 50, 1000) - 0.1) < 1e-9);
    assert(std::abs(stepSched->getRate(0.1, 150, 1000) - 0.05) < 1e-9);
    assert(std::abs(stepSched->getRate(0.1, 250, 1000) - 0.025) < 1e-9);

    auto cosSched = SchedulerFactory::create("cosine", 100, 0.5, 0.001);
    assert(cosSched->name() == "Cosine Annealing");
    assert(std::abs(cosSched->getRate(0.1, 0, 1000) - 0.1) < 1e-9);
    assert(std::abs(cosSched->getRate(0.1, 1000, 1000) - 0.001) < 1e-9);
    std::cout << "PASS lr-scheduler-strategy\n";
}

int main() {
    testNormalizers();
    testWeightInits();
    testValidatorRules();
    testDatasetSources();
    testTargetSelectorChain();
    testSplitOrders();
    testVisitorTraversal();
    testLRSchedulers();
    std::cout << "ALL PATTERN CHECKS PASS\n";
    return 0;
}
