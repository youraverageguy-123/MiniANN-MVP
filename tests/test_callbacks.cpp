// Tests for EarlyStopping, CallbackList, FileLogger and TrainingConfig::validate.
// Run from the project root (the integration test reads data/iris_small.csv).
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "miniann/miniann.hpp"
#include "miniann/callbacks.hpp"

using namespace miniann;

// Fake history builder: lets us feed EarlyStopping any loss curve we like.
static void feed(EarlyStopping& es, const std::vector<double>& losses, bool asValidation) {
    TrainingHistory h;
    for (std::size_t i = 0; i < losses.size(); ++i) {
        (asValidation ? h.validationLoss : h.trainLoss).push_back(losses[i]);
        es.onEpoch(int(i) + 1, h);
        if (es.shouldStop()) break;
    }
}

static void testImprovingNeverStops() {
    EarlyStopping es(3);
    feed(es, {1.0, 0.9, 0.8, 0.7, 0.6, 0.5}, true);
    assert(!es.shouldStop());
    assert(es.bestEpoch() == 6 && es.bestLoss() == 0.5);
    std::cout << "PASS improving-never-stops\n";
}

static void testStopsAfterPatience() {
    EarlyStopping es(3);
    // best at epoch 3 (0.5); epochs 4,5,6 are worse -> stop exactly at epoch 6
    feed(es, {1.0, 0.7, 0.5, 0.6, 0.6, 0.7, 0.4, 0.3}, true);
    assert(es.shouldStop());
    assert(es.stoppedAtEpoch() == 6);
    assert(es.bestEpoch() == 3 && es.bestLoss() == 0.5);
    std::cout << "PASS stops-after-patience\n";
}

static void testMinDeltaIgnoresTinyGains() {
    EarlyStopping es(2, 0.01);
    // gains of 0.001 are below minDelta -> not "improvements"
    feed(es, {1.0, 0.5, 0.499, 0.498, 0.497}, true);
    assert(es.shouldStop() && es.stoppedAtEpoch() == 4);
    assert(es.bestEpoch() == 2);
    std::cout << "PASS min-delta\n";
}

static void testFallsBackToTrainLoss() {
    EarlyStopping es(2);
    feed(es, {1.0, 0.5, 0.6, 0.6}, false); // no validation curve supplied
    assert(es.shouldStop() && !es.usedValidation());
    std::cout << "PASS falls-back-to-train-loss\n";
}

static void testNaNEventuallyStops() {
    EarlyStopping es(2);
    feed(es, {1.0, std::nan(""), std::nan(""), std::nan("")}, true);
    assert(es.shouldStop()); // a diverged run must not train forever
    std::cout << "PASS nan-stops\n";
}

static void testResetAndBadArgs() {
    EarlyStopping es(1);
    feed(es, {1.0, 2.0}, true);
    assert(es.shouldStop());
    es.reset();
    assert(!es.shouldStop() && es.bestEpoch() == 0 && es.stoppedAtEpoch() == 0);
    bool a = false, b = false;
    try { EarlyStopping bad(0); } catch (const std::invalid_argument&) { a = true; }
    try { EarlyStopping bad(5, -0.1); } catch (const std::invalid_argument&) { b = true; }
    assert(a && b);
    std::cout << "PASS reset-and-bad-args\n";
}

// Counts calls, and can be told to request a stop.
struct Probe : TrainingCallback {
    int calls = 0;
    bool stop = false;
    void onEpoch(int, const TrainingHistory&) override { ++calls; }
    bool shouldStop() const override { return stop; }
};

static void testCallbackList() {
    Probe a, b;
    CallbackList list;
    list.add(&a);
    list.add(&b);
    assert(list.size() == 2);
    TrainingHistory h;
    list.onEpoch(1, h);
    list.onEpoch(2, h);
    assert(a.calls == 2 && b.calls == 2);   // forwarded to everyone
    assert(!list.shouldStop());
    b.stop = true;
    assert(list.shouldStop());              // ANY callback can stop training
    bool threw = false;
    try { list.add(nullptr); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    std::cout << "PASS callback-list\n";
}

static void testFileLogger() {
    const char* path = "test_filelogger.log";
    {
        FileLogger fl(path, LogLevel::Warn);
        ILogger& base = fl; // used through the interface -> polymorphism
        base.log(LogLevel::Info, "hidden");     // below minLevel -> dropped
        base.log(LogLevel::Warn, "careful");
        base.log(LogLevel::Error, "broken");
    }
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string text = ss.str();
    assert(text.find("hidden") == std::string::npos);
    assert(text.find("[WARN] careful") != std::string::npos);
    assert(text.find("[ERROR] broken") != std::string::npos);
    in.close();
    std::remove(path);

    bool threw = false;
    try { FileLogger bad("/nonexistent_dir_xyz/out.log"); } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    std::cout << "PASS file-logger\n";
}

static void testConfigValidation() {
    std::mt19937 rng(1);
    NeuralNetwork net;
    net.addLayer(Layer(3, 2, ActivationFactory::create("tanh"), rng));
    net.addLayer(Layer(1, 3, ActivationFactory::create("sigmoid"), rng));
    MSELoss loss;
    SGD opt(0.5);
    Trainer t(net, loss, opt);
    Dataset d = makeXorGate();

    TrainingConfig c;
    c.epochs = 2;
    c.logEvery = 0;      // used to crash with a divide-by-zero (SIGFPE)
    bool threw = false;
    try { t.fit(d, nullptr, c); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);

    c.logEvery = 1;
    c.epochs = 0;
    threw = false;
    try { t.fit(d, nullptr, c); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);

    c.epochs = 2;        // a valid config still trains normally
    assert(t.fit(d, nullptr, c).trainLoss.size() == 2);
    std::cout << "PASS config-validation\n";
}

// End-to-end: real network + real Trainer + EarlyStopping inside a CallbackList.
static void testIntegrationWithTrainer() {
    Dataset raw = Dataset::loadCSV("data/iris_small.csv", true, 4, 1);
    Dataset norm = normalizeMaxAbs(raw);
    auto [trRaw, vaRaw, teRaw] = norm.splitTrainValTest(0.6, 0.2, 42);
    Dataset tr = oneHotEncode(trRaw, 3), va = oneHotEncode(vaRaw, 3);
    (void)teRaw;

    std::mt19937 rng(42);
    NeuralNetwork net;
    net.addLayer(Layer(6, 4, ActivationFactory::create("tanh"), rng));
    net.addLayer(Layer(3, 6, ActivationFactory::create("sigmoid"), rng));
    MSELoss loss;
    Adam opt(0.01);
    Trainer trainer(net, loss, opt);

    EarlyStopping es(25, 1e-5);
    Probe probe;
    CallbackList cbs;
    cbs.add(&probe);
    cbs.add(&es);

    TrainingConfig cfg;
    cfg.epochs = 20000;       // far more than needed
    cfg.batchSize = 8;
    cfg.logEvery = 1000;
    TrainingHistory h = trainer.fit(tr, &va, cfg, &cbs);

    assert(es.stopped());
    assert(int(h.trainLoss.size()) == es.stoppedAtEpoch());  // loop really broke
    assert(int(h.trainLoss.size()) < cfg.epochs);            // ...long before the limit
    assert(probe.calls == int(h.trainLoss.size()));          // other callback saw every epoch
    assert(es.stoppedAtEpoch() - es.bestEpoch() == es.patience());
    std::cout << "PASS integration (stopped at epoch " << es.stoppedAtEpoch()
              << " of " << cfg.epochs << ", best epoch " << es.bestEpoch() << ")\n";
}

int main() {
    testImprovingNeverStops();
    testStopsAfterPatience();
    testMinDeltaIgnoresTinyGains();
    testFallsBackToTrainLoss();
    testNaNEventuallyStops();
    testResetAndBadArgs();
    testCallbackList();
    testFileLogger();
    testConfigValidation();
    testIntegrationWithTrainer();
    std::cout << "ALL CALLBACK CHECKS PASS\n";
    return 0;
}
