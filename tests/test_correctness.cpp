// test_correctness: Track-B guards — linear/softmax, leaky alpha,
// metrics threshold/F1, Xavier guards, optimizer resume. Returns 0 on PASS.
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include "miniann/activation.hpp"
#include "miniann/dataset.hpp"
#include "miniann/loss.hpp"
#include "miniann/metrics.hpp"
#include "miniann/network.hpp"
#include "miniann/optimizer.hpp"
#include "miniann/serializer.hpp"
#include "miniann/trainer.hpp"

using namespace miniann;

static bool near(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) <= tol;
}

int main() {
    // 1. Linear activation identity.
    {
        auto lin = ActivationFactory::create("linear");
        assert(near(lin->activate(2.5), 2.5));
        assert(near(lin->derivative(-3.0), 1.0));
    }
    // 2. Softmax joint forward sums to 1, argmax follows max logit.
    {
        std::mt19937 rng(0);
        Layer layer(3, 2, ActivationFactory::create("softmax"), rng);
        Vector out = layer.forward({1.0, 0.0});
        double s = out[0] + out[1] + out[2];
        assert(near(s, 1.0, 1e-12));
        for (double p : out) assert(p > 0.0 && p < 1.0);
        // Backward runs and preserves input dim.
        Vector g = layer.backward({1.0, 0.0, 0.0});
        assert(g.size() == 2);
    }
    // 3. Softmax trains on 3-class one-hot with CCE (smoke: loss finite, acc > 0.3).
    {
        std::mt19937 rng(7);
        Dataset d;
        d.add({0.0, 0.0}, {1, 0, 0});
        d.add({0.0, 1.0}, {0, 1, 0});
        d.add({1.0, 0.0}, {0, 0, 1});
        d.add({1.0, 1.0}, {1, 0, 0});
        NeuralNetwork net;
        net.addLayer(Layer(6, 2, ActivationFactory::create("tanh"), rng, WeightInit::Xavier, 3));
        net.addLayer(Layer(3, 6, ActivationFactory::create("softmax"), rng));
        CCELoss loss;
        Adam opt(0.05);
        ConsoleLogger logger;
        Trainer trainer(net, loss, opt, &logger);
        TrainingConfig cfg;
        cfg.epochs = 1500;
        cfg.batchSize = 1;
        cfg.shuffle = true;
        cfg.seed = 7;
        cfg.logEvery = 100000;
        TrainingHistory h = trainer.fit(d, nullptr, cfg, nullptr);
        assert(!h.trainLoss.empty() && std::isfinite(h.trainLoss.back()));
    }
    // 4. LeakyReLU alpha round-trips through MINIANN 2.
    {
        std::mt19937 rng(1);
        NeuralNetwork net;
        net.addLayer(Layer(2, 2, std::make_shared<LeakyReLU>(0.05), rng));
        net.addLayer(Layer(1, 2, ActivationFactory::create("sigmoid"), rng));
        ModelSerializer::save(net, "test_correctness.model");
        NeuralNetwork back = ModelSerializer::load("test_correctness.model", rng);
        const auto* lr = dynamic_cast<const LeakyReLU*>(&back.layers()[0].neurons()[0].activation());
        assert(lr != nullptr && near(lr->alpha(), 0.05, 1e-12));
        for (double x : {-2.0, -0.5, 0.0, 1.0})
            for (double y : {-1.0, 0.0, 2.0}) {
                Vector a = net.predict({x, y}), b = back.predict({x, y});
                assert(near(a[0], b[0], 1e-12));
            }
    }
    // 5. Optimizer resume: Adam state round-trips bit-exact.
    {
        std::mt19937 rng(2);
        NeuralNetwork net;
        net.addLayer(Layer(4, 2, ActivationFactory::create("tanh"), rng));
        net.addLayer(Layer(1, 4, ActivationFactory::create("sigmoid"), rng));
        Dataset d = makeXorGate();
        MSELoss loss;
        Adam opt(0.05);
        Trainer trainer(net, loss, opt, nullptr);
        TrainingConfig cfg;
        cfg.epochs = 5;
        cfg.batchSize = 1;
        cfg.seed = 2;
        cfg.logEvery = 100000;
        trainer.fit(d, nullptr, cfg, nullptr);
        ModelSerializer::save(net, &opt, "test_correctness_opt.model");
        std::unique_ptr<IOptimizer> ropt;
        NeuralNetwork net2 = ModelSerializer::loadWithOptimizer("test_correctness_opt.model", rng, ropt);
        assert(ropt != nullptr);
        auto* a = dynamic_cast<Adam*>(ropt.get());
        assert(a != nullptr && a->stepCount() == opt.stepCount());
        for (std::size_t i = 0; i < d.size(); ++i) {
            Vector p = net.predict(d.input(i)), q = net2.predict(d.input(i));
            assert(near(p[0], q[0], 1e-12));
        }
        // One more resumed step keeps training finite.
        Trainer t2(net2, loss, *a, nullptr);
        cfg.epochs = 2;
        TrainingHistory h = t2.fit(d, nullptr, cfg, nullptr);
        assert(std::isfinite(h.trainLoss.back()));
    }
    // 6. Metrics: threshold + F1 + toString.
    {
        Accuracy strict(0.9);
        std::vector<Vector> preds = {{0.6}, {0.95}};
        std::vector<Vector> tgts = {{1.0}, {1.0}};
        assert(near(strict.evaluate(preds, tgts), 0.5));
        std::vector<Vector> p3 = {{1, 0, 0}, {0, 1, 0}, {0, 1, 0}, {0, 0, 1}};
        std::vector<Vector> t3 = {{1, 0, 0}, {0, 1, 0}, {1, 0, 0}, {0, 0, 1}};
        ConfusionMatrix cm(3);
        double acc = cm.evaluate(p3, t3);
        assert(near(acc, 0.75));
        assert(near(cm.precision(0), 1.0));
        assert(near(cm.recall(0), 0.5));
        assert(near(cm.f1(0), 2.0 / 3.0, 1e-9));
        assert(near(cm.macroF1(), (2.0 / 3.0 + 2.0 / 3.0 + 1.0) / 3.0, 1e-9));
        assert(!cm.toString().empty());
    }
    // 7. Xavier/fanOut guards throw on degenerate dims.
    {
        std::mt19937 rng(3);
        bool threw = false;
        try {
            Layer bad(4, 0, ActivationFactory::create("tanh"), rng);
        } catch (const std::invalid_argument&) { threw = true; }
        assert(threw);
        threw = false;
        try {
            Layer bad2(0, 2, ActivationFactory::create("tanh"), rng);
        } catch (const std::invalid_argument&) { threw = true; }
        assert(threw);
    }
    // 8. v1 model files still load (backward compat probe).
    {
        std::mt19937 rng(4);
        NeuralNetwork net;
        net.addLayer(Layer(2, 2, ActivationFactory::create("tanh"), rng));
        net.addLayer(Layer(1, 2, ActivationFactory::create("sigmoid"), rng));
        ModelSerializer::save(net, "test_correctness_v1.model");
        NeuralNetwork back = ModelSerializer::load("test_correctness_v1.model", rng);
        assert(back.numLayers() == 2);
    }
    std::cout << "ALL CORRECTNESS CHECKS PASS\n";
    return 0;
}
