// playground: the interactive layer of the revised spec (§28).
// No C++ edits needed to experiment: pick dataset, architecture, activations,
// loss, optimizer, and training parameters at prompts, then watch it learn
// (loss/accuracy curves, boundary map, weights) and query predictions.
#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>
#include <memory>
#include <algorithm>
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

static std::string line(const std::string& prompt, const std::string& dflt) {
    std::cout << prompt << " [" << dflt << "]: ";
    std::string s;
    std::getline(std::cin, s);
    return s.empty() ? dflt : s;
}

static Dataset oneHot(const Dataset& raw, int k) {
    Dataset d;
    for (std::size_t i = 0; i < raw.size(); ++i) {
        int c = int(raw.target(i)[0]);
        Vector t(std::size_t(k), 0.0);
        if (c >= 0 && c < k) t[std::size_t(c)] = 1.0;
        d.add(raw.input(i), t);
    }
    return d;
}

static Dataset normalizeMaxAbs(Dataset d) {
    double m = 0.0;
    for (std::size_t i = 0; i < d.size(); ++i)
        for (double v : d.input(i)) m = std::max(m, std::abs(v));
    if (m == 0.0) m = 1.0;
    Dataset out;
    for (std::size_t i = 0; i < d.size(); ++i) {
        Vector x = d.input(i);
        for (auto& v : x) v /= m;
        out.add(x, d.target(i));
    }
    return out;
}

int main() {
    std::cout << "---------------------------------------------------\n"
              << "                    MiniANN Playground\n"
              << "         Build the Network. Train It. See It Learn.\n"
              << "---------------------------------------------------\n";
    try {
        // ---- dataset ----
        std::cout << "\nDataset: 1=AND 2=OR 3=XOR 4=Iris 5=CSV file\n";
        std::string pick = line("choice", "3");
        Dataset raw;
        int numClasses = 2;
        if (pick == "1") raw = makeAndGate();
        else if (pick == "2") raw = makeOrGate();
        else if (pick == "3") raw = makeXorGate();
        else if (pick == "4") {
            raw = Dataset::loadCSV("data/iris_small.csv", true, 4, 1);
            raw = normalizeMaxAbs(raw);
            raw = oneHot(raw, 3);
            numClasses = 3;
            std::cout << "iris: normalized by max-abs, labels one-hot encoded (3 classes)\n";
        } else {
            std::string path = line("csv path", "data/iris_small.csv");
            std::string hs = line("has header? y/n", "y");
            int tcol = std::stoi(line("target column index", "4"));
            int tdim = std::stoi(line("target dim", "1"));
            raw = Dataset::loadCSV(path, hs == "y", tcol, tdim);
            if (tdim == 1 && line("int class labels needing one-hot? y/n", "n") == "y") {
                numClasses = std::stoi(line("num classes", "3"));
                raw = normalizeMaxAbs(raw);
                raw = oneHot(raw, numClasses);
            } else if (raw.target(0).size() > 1) {
                numClasses = int(raw.target(0).size());
            }
        }
        std::cout << "loaded: " << raw.size() << " samples, in=" << raw.input(0).size()
                  << " out=" << raw.target(0).size() << "\n";

        double split = std::stod(line("train fraction 0..1", "0.8"));
        std::mt19937 srng(42);
        auto [trFull, teFull] = raw.split(split, srng);

        // ---- architecture ----
        std::cout << "\nArchitecture as sizes, e.g. \"2 8 8 1\" (first = inputs, last = outputs)\n";
        std::vector<int> sizes;
        { std::istringstream ss(line("sizes", raw.input(0).size() == 2 ? "2 8 8 1" : "4 6 3")); int v;
          while (ss >> v) sizes.push_back(v); }
        if (sizes.size() < 2) throw std::invalid_argument("need >= 2 sizes");
        if (sizes.front() != int(raw.input(0).size()) || sizes.back() != int(raw.target(0).size()))
            throw std::invalid_argument("sizes must start with input dim and end with output dim");
        std::cout << "one activation per layer (" << sizes.size() - 1 << "): "
                     "sigmoid|tanh|relu|leaky_relu|swish\n";
        std::vector<std::string> acts;
        { std::istringstream ss(line("activations", sizes.size() == 4 ? "tanh tanh sigmoid" : "tanh sigmoid"));
          std::string a; while (ss >> a) acts.push_back(a); }
        if (acts.size() != sizes.size() - 1) throw std::invalid_argument("activation count mismatch");
        for (auto& a : acts) ActivationFactory::create(a); // validate early

        // ---- loss / optimizer / training ----
        std::string lossName = line("loss mse|bce|cce", raw.target(0).size() == 1 ? "mse" : "cce");
        auto loss = LossFactory::create(lossName);
        if (lossName == "bce" && raw.target(0).size() != 1)
            throw std::invalid_argument("bce needs a 1-output network");
        std::string optName = line("optimizer sgd|momentum|adam", "adam");
        double lr = std::stod(line("learning rate", optName == "adam" ? "0.01" : "0.1"));
        auto opt = OptimizerFactory::create(optName, lr);
        int epochs = std::stoi(line("epochs", "1500"));
        int batch = std::stoi(line("batch size (0=full)", "8"));
        unsigned seed = unsigned(std::stoul(line("seed", "42")));

        // ---- build ----
        std::mt19937 rng(seed);
        NeuralNetwork net;
        for (std::size_t l = 1; l < sizes.size(); ++l)
            net.addLayer(Layer(std::size_t(sizes[l]), std::size_t(sizes[l - 1]),
                               ActivationFactory::create(acts[l - 1]), rng,
                               cli::defaultInitFor(acts[l - 1])));
        NetworkGraph ng(net);
        std::cout << "\n--- " << ng.title() << " ---\n" << ng.render();

        ConsoleLogger logger;
        Trainer trainer(net, *loss, *opt, &logger);
        TrainingConfig cfg;
        cfg.epochs = epochs; cfg.batchSize = std::size_t(batch);
        cfg.shuffle = true; cfg.seed = seed; cfg.logEvery = std::max(1, epochs / 10);
        std::cout << "\n[TRAIN] live view below (redraws as it learns)\n";
        LiveConsole live(net, trFull.input(0), "MiniANN live", epochs,
                         std::max(1, epochs / 60));
        TrainingHistory h = trainer.fit(trFull, &teFull, cfg, &live);
        CSVLossExporter::exportHistory(h, "playground_loss.csv");
        ModelSerializer::save(net, "playground.model");
        std::cout << "saved playground.model + playground_loss.csv\n";

        HtmlReport report("MiniANN playground run");
        ReportSeries rs;
        rs.name = optName + " lr=" + std::to_string(lr);
        rs.loss = h.trainLoss;
        rs.acc = h.trainAcc;
        report.addSeries(rs);
        report.setBoundary(&net, &trFull);
        report.addPre("network structure", ng.render());
        WeightsTable wt(net);
        report.addPre("learned weights", wt.render());
        report.save("playground_report.html");
        std::cout << "HTML report -> playground_report.html (open in a browser)\n";

        // ---- visuals ----
        LossCurve lc(h);
        AccuracyCurve ac(h);
        std::cout << "\n--- " << lc.title() << " ---\n" << lc.render();
        std::cout << "\n--- " << ac.title() << " ---\n" << ac.render();
        if (raw.input(0).size() == 2 && raw.target(0).size() == 1) {
            BoundaryMap bm(net, trFull);
            std::cout << "\n--- " << bm.title() << " ---\n" << bm.render();
        }
        std::cout << "\n--- " << wt.title() << " ---\n" << wt.render();

        // ---- evaluation ----
        auto collect = [&](const Dataset& d, std::vector<Vector>& p, std::vector<Vector>& t) {
            for (std::size_t i = 0; i < d.size(); ++i) {
                p.push_back(net.predict(d.input(i)));
                t.push_back(d.target(i));
            }
        };
        std::vector<Vector> pTe, tTe;
        collect(teFull, pTe, tTe);
        Accuracy acc;
        std::cout << "test accuracy=" << acc.evaluate(pTe, tTe) * 100.0 << "%\n";
        ConfusionMatrix cm{std::size_t(numClasses)};
        cm.evaluate(pTe, tTe);
        std::cout << "confusion (rows=true, cols=pred):\n";
        cm.print();

        // ---- prediction REPL ----
        std::cout << "\nPredict: type " << raw.input(0).size()
                  << " space-separated features, or q to quit\n";
        std::string s;
        while (true) {
            std::cout << "> ";
            if (!std::getline(std::cin, s) || s == "q" || s == "quit") break;
            std::istringstream ss(s);
            Vector x;
            double v;
            while (ss >> v) x.push_back(v);
            if (x.size() != raw.input(0).size()) {
                std::cout << "need " << raw.input(0).size() << " values\n";
                continue;
            }
            Vector p = net.predict(x);
            if (p.size() == 1) {
                int cls = p[0] >= 0.5 ? 1 : 0;
                double conf = cls ? p[0] : 1.0 - p[0];
                std::cout << "class " << cls << " confidence "
                          << std::fixed << std::setprecision(1) << conf * 100.0 << "%\n";
            } else {
                std::size_t b = 0;
                for (std::size_t k = 1; k < p.size(); ++k) if (p[k] > p[b]) b = k;
                std::cout << "class " << b << " confidence "
                          << std::fixed << std::setprecision(1) << p[b] * 100.0 << "%\n";
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "playground error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
