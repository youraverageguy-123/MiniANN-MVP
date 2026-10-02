// xor_demo: proves non-linear learning. Defaults: 2-4-1 tanh/sigmoid, SGD lr=0.5.
// The user picks descent method, training setup, and hidden activation:
//   xor_demo.exe --opt sgd|momentum|adam --lr 0.5 --act tanh|relu|leaky_relu|swish
//                --epochs 10000 --batch 1 --seed 42 --hidden 4 --live|--no-live
#include <iostream>
#include <iomanip>
#include <random>
#include <memory>
#include <vector>
#include "miniann/network.hpp"
#include "miniann/dataset.hpp"
#include "miniann/loss.hpp"
#include "miniann/optimizer.hpp"
#include "miniann/trainer.hpp"
#include "miniann/metrics.hpp"
#include "miniann/logger.hpp"
#include "miniann/serializer.hpp"
#include "miniann/cli.hpp"
#include "miniann/visualizer.hpp"

using namespace miniann;

int main(int argc, char** argv) {
    cli::Options o;
    o.lr = 0.5; o.epochs = 10000; o.batch = 1; o.seed = 42; o.hidden = 4;
    try {
        o = cli::parse(argc, argv, o);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    std::mt19937 rng(o.seed);
    Dataset data = makeXorGate();
    data.validate();

    WeightInit init = cli::defaultInitFor(o.hiddenAct); // He for relu-family
    NeuralNetwork net;
    net.addLayer(Layer(std::size_t(o.hidden), 2,
                       ActivationFactory::create(o.hiddenAct), rng, init));
    net.addLayer(Layer(1, std::size_t(o.hidden),
                       ActivationFactory::create("sigmoid"), rng, WeightInit::Xavier));

    MSELoss loss;
    auto opt = cli::makeOptimizer(o);
    ConsoleLogger logger;
    Trainer trainer(net, loss, *opt, &logger);

    TrainingConfig cfg;
    cfg.epochs = o.epochs;
    cfg.batchSize = std::size_t(o.batch);
    cfg.shuffle = true;
    cfg.seed = o.seed;
    cfg.logEvery = std::max(1, o.epochs / 5);

    std::cout << "XOR 2-" << o.hidden << "-1 " << o.hiddenAct << "/sigmoid, "
              << o.optimizer << " lr=" << o.lr << " batch=" << o.batch << "\n";
    // Live monitor redraws plots + neuron firing as epochs stream in.
    LiveConsole live(net, {1.0, 0.0}, "MiniANN live -- XOR", o.epochs,
                     std::max(1, o.epochs / 120));
    TrainingHistory hist = trainer.fit(data, nullptr, cfg, o.live ? &live : nullptr);
    CSVLossExporter::exportHistory(hist, "xor_loss.csv");

    // Evaluate
    std::vector<Vector> preds, targets;
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

    // Visuals: every visual is an IVisualizer, printed polymorphically.
    std::vector<std::unique_ptr<IVisualizer>> visuals;
    visuals.push_back(std::make_unique<BoundaryMap>(net, data));
    visuals.push_back(std::make_unique<LossCurve>(hist));
    for (auto& v : visuals) {
        std::cout << "--- " << v->title() << " ---\n" << v->render();
    }

    ModelSerializer::save(net, "xor.model");
    NeuralNetwork net2 = ModelSerializer::load("xor.model", rng);    double maxDiff = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        Vector a = net.predict(data.input(i));
        Vector b = net2.predict(data.input(i));
        maxDiff = std::max(maxDiff, std::abs(a[0] - b[0]));
    }
    std::cout << "serialization round-trip max|y_orig - y_loaded|=" << maxDiff << "\n";

    HtmlReport report("XOR 2-" + std::to_string(o.hidden) + "-1 " + o.hiddenAct +
                      "/sigmoid, " + o.optimizer);
    ReportSeries rs;
    rs.name = o.optimizer + " lr=" + std::to_string(o.lr);
    rs.loss = hist.trainLoss;
    rs.acc = hist.trainAcc;
    report.addSeries(rs);
    report.setBoundary(&net, &data);
    NetworkGraph ng(net);
    report.addPre("network structure", ng.render());
    report.save("xor_report.html");
    std::cout << "HTML report -> xor_report.html (open in a browser)\n";
    return 0;
}
