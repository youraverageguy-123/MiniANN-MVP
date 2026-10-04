// early_stop_demo: trains Iris with a HUGE epoch budget and lets EarlyStopping
// end the run once validation loss stops improving. Also shows FileLogger
// (writes early_stop_training.log) and CallbackList (two callbacks at once).
// Run from the project root:  early_stop_demo.exe
#include <iostream>
#include <random>
#include "miniann/miniann.hpp"
#include "miniann/callbacks.hpp"

using namespace miniann;

// A tiny extra callback, just to prove CallbackList forwards to everyone:
// prints one line every N epochs.
class ProgressPrinter : public TrainingCallback {
public:
    explicit ProgressPrinter(int every) : every_(every) {}
    void onEpoch(int epoch, const TrainingHistory& h) override {
        if (epoch % every_ == 0)
            std::cout << "  epoch " << epoch << "  val_loss=" << h.validationLoss.back() << "\n";
    }
private:
    int every_;
};

int main() {
    Dataset raw;
    try {
        raw = Dataset::loadCSV("data/iris_small.csv", true, 4, 1);
    } catch (const std::exception& e) {
        std::cerr << "early_stop_demo: " << e.what() << " (run from the project root)\n";
        return 1;
    }
    auto [trRaw, vaRaw, teRaw] = normalizeMaxAbs(raw).splitTrainValTest(0.6, 0.2, 42);
    Dataset tr = oneHotEncode(trRaw, 3), va = oneHotEncode(vaRaw, 3), te = oneHotEncode(teRaw, 3);

    std::mt19937 rng(42);
    NeuralNetwork net;
    net.addLayer(Layer(6, 4, ActivationFactory::create("tanh"), rng));
    net.addLayer(Layer(3, 6, ActivationFactory::create("sigmoid"), rng));

    MSELoss loss;
    Adam opt(0.01);
    FileLogger logger("early_stop_training.log");        // ILogger -> file instead of screen
    Trainer trainer(net, loss, opt, &logger);

    EarlyStopping stopper(/*patience=*/100, /*minDelta=*/1e-4);
    ProgressPrinter printer(50);
    CallbackList callbacks;                               // Composite: one slot, two callbacks
    callbacks.add(&printer);
    callbacks.add(&stopper);

    TrainingConfig cfg;
    cfg.epochs = 20000;                                   // budget we hope NOT to use up
    cfg.batchSize = 8;
    cfg.logEvery = 50;

    std::cout << "Training Iris, budget = " << cfg.epochs << " epochs, patience = "
              << stopper.patience() << "\n";
    TrainingHistory h = trainer.fit(tr, &va, cfg, &callbacks);

    std::cout << "\nEarly stopping " << (stopper.stopped() ? "TRIGGERED" : "did not trigger") << "\n"
              << "  epochs actually run : " << h.trainLoss.size() << " / " << cfg.epochs << "\n"
              << "  best epoch          : " << stopper.bestEpoch()
              << " (val_loss=" << stopper.bestLoss() << ")\n"
              << "  watched             : " << (stopper.usedValidation() ? "validation" : "training")
              << " loss\n";

    std::vector<Vector> p, t;
    for (std::size_t i = 0; i < te.size(); ++i) {
        p.push_back(net.predict(te.input(i)));
        t.push_back(te.target(i));
    }
    std::cout << "  test accuracy       : " << Accuracy().evaluate(p, t) * 100.0 << "%\n"
              << "Log written to " << logger.path() << "\n";
    return 0;
}
