// compare_demo (§13 of the revised spec): same network + dataset, three
// descent methods, one table. All values come from actual training runs.
#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>
#include <memory>
#include <vector>
#include "miniann/network.hpp"
#include "miniann/dataset.hpp"
#include "miniann/loss.hpp"
#include "miniann/optimizer.hpp"
#include "miniann/trainer.hpp"
#include "miniann/metrics.hpp"
#include "miniann/visualizer.hpp"

using namespace miniann;

struct Row {
    std::string opt;
    double lr;
    double finalLoss;
    double acc;
    TrainingHistory hist;
};

int main() {
    const std::vector<std::pair<std::string, double>> cfgs = {
        {"sgd", 0.5}, {"momentum", 0.1}, {"adam", 0.01}};
    std::vector<Row> rows;

    for (auto& [name, lr] : cfgs) {
        std::mt19937 rng(42); // same init => fair comparison
        Dataset data = makeXorGate();

        NeuralNetwork net;
        net.addLayer(Layer(8, 2, ActivationFactory::create("tanh"), rng, WeightInit::Xavier));
        net.addLayer(Layer(8, 8, ActivationFactory::create("tanh"), rng, WeightInit::Xavier));
        net.addLayer(Layer(1, 8, ActivationFactory::create("sigmoid"), rng, WeightInit::Xavier));

        MSELoss loss;
        auto opt = OptimizerFactory::create(name, lr);
        Trainer trainer(net, loss, *opt, nullptr);
        TrainingConfig cfg;
        cfg.epochs = 3000; cfg.batchSize = 1; cfg.seed = 42; cfg.logEvery = 3001;
        TrainingHistory h = trainer.fit(data, nullptr, cfg);

        std::vector<Vector> preds, targets;
        for (std::size_t i = 0; i < data.size(); ++i) {
            preds.push_back(net.predict(data.input(i)));
            targets.push_back(data.target(i));
        }
        Accuracy acc;
        rows.push_back({name, lr, h.trainLoss.back(), acc.evaluate(preds, targets), h});
    }

    std::cout << "Dataset: XOR   Architecture: 2 -> 8 -> 8 -> 1 (tanh/tanh/sigmoid)\n\n";
    std::cout << "| Optimizer | LR   | Epochs | Final Loss | Accuracy |\n"
              << "|-----------|------|--------|------------|----------|\n";
    for (auto& r : rows) {
        std::ostringstream lr, loss, acc;
        lr << r.lr;
        loss << std::setprecision(4) << std::scientific << r.finalLoss;
        acc << std::fixed << std::setprecision(1) << r.acc * 100.0 << "%";
        std::cout << "| " << std::left << std::setw(9) << r.opt << " | "
                  << std::setw(4) << lr.str() << " | 3000   | "
                  << std::setw(10) << loss.str() << " | "
                  << std::setw(8) << acc.str() << " |\n";
    }

    for (auto& r : rows) {
        std::cout << "\n--- " << r.opt << " loss curve ---\n";
        LossCurve lc(r.hist, 50, 8);
        std::cout << lc.render();
    }

    HtmlReport report("XOR optimizer comparison (2-8-8-1 tanh/tanh/sigmoid)");
    for (auto& r : rows) {
        ReportSeries rs;
        rs.name = r.opt;
        rs.loss = r.hist.trainLoss;
        rs.acc = r.hist.trainAcc;
        report.addSeries(rs);
    }
    report.save("compare_report.html");
    std::cout << "\nHTML report -> compare_report.html (open in a browser)\n";
    return 0;
}
