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
#include "miniann/report.hpp"

using namespace miniann;

static std::string trimStr(std::string s) {
    const char* ws = " \t\r\n";
    s.erase(0, s.find_first_not_of(ws));
    if (!s.empty()) s.erase(s.find_last_not_of(ws) + 1);
    return s;
}

static std::string line(const std::string& prompt, const std::string& dflt) {
    std::cout << prompt << " [" << dflt << "]: ";
    std::string s;
    std::getline(std::cin, s);
    s = trimStr(std::move(s));
    return s.empty() ? dflt : s;
}

static int parseInt(const std::string& raw, const char* what) {
    try {
        std::size_t pos = 0;
        int v = std::stoi(raw, &pos);
        if (pos != trimStr(raw).size()) throw std::invalid_argument("trailing text");
        return v;
    } catch (...) {
        throw std::invalid_argument(std::string(what) + " must be an integer (got '" + raw + "')");
    }
}

static double parseDouble(const std::string& raw, const char* what) {
    try {
        std::size_t pos = 0;
        double v = std::stod(raw, &pos);
        if (pos != trimStr(raw).size()) throw std::invalid_argument("trailing text");
        return v;
    } catch (...) {
        throw std::invalid_argument(std::string(what) + " must be a number (got '" + raw + "')");
    }
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
            int tcol = parseInt(line("target column index (-1 = last col)", "4"),
                                "target column index (-1 = last column)");
            int tdim = parseInt(line("target dim", "1"), "target dim");
            if (tdim < 1) throw std::invalid_argument("target dim must be >= 1");
            raw = Dataset::loadCSV(path, hs == "y" || hs == "Y", tcol, tdim);
            if (tdim == 1 && line("int class labels needing one-hot? y/n", "n") == "y") {
                numClasses = parseInt(line("num classes", "3"), "num classes");
                if (numClasses < 2) throw std::invalid_argument("num classes must be >= 2");
                raw = normalizeMaxAbs(raw);
                raw = oneHot(raw, numClasses);
            } else if (raw.target(0).size() > 1) {
                numClasses = int(raw.target(0).size());
            }
        }
        std::cout << "loaded: " << raw.size() << " samples, in=" << raw.input(0).size()
                  << " out=" << raw.target(0).size() << "\n";

        double split = parseDouble(line("train fraction 0..1", "0.8"), "train fraction");
        if (!(split > 0.0 && split < 1.0))
            throw std::invalid_argument("train fraction must be in (0,1), e.g. 0.8");
        {
            std::size_t nTrain = std::size_t(double(raw.size()) * split);
            std::size_t nTest = raw.size() - nTrain;
            if (nTrain < 2 || nTest < 1)
                throw std::invalid_argument(
                    "train fraction " + std::to_string(split) + " leaves " +
                    std::to_string(nTrain) + " train / " +
                    std::to_string(nTest) + " test samples; " +
                    "need >=2 train and >=1 test (with only " + std::to_string(raw.size()) +
                    " samples try 0.5 or 0.75)");
        }
        std::mt19937 srng(42);
        auto [trFull, teFull] = raw.split(split, srng);

        // ---- architecture ----
        std::cout << "\nArchitecture as sizes, e.g. \"2 8 8 1\" (first = inputs, last = outputs)\n";
        std::vector<int> sizes;
        { std::istringstream ss(line("sizes", raw.input(0).size() == 2 ? "2 8 8 1" : "4 6 3")); int v;
          while (ss >> v) sizes.push_back(v); }
        if (sizes.size() < 2) throw std::invalid_argument("need >= 2 sizes, e.g. \"2 8 8 1\"");
        for (int v : sizes)
            if (v < 1) throw std::invalid_argument("all sizes must be >= 1");
        if (sizes.front() != int(raw.input(0).size()) || sizes.back() != int(raw.target(0).size()))
            throw std::invalid_argument(
                "sizes must start with input dim " + std::to_string(raw.input(0).size()) +
                " and end with output dim " + std::to_string(raw.target(0).size()) +
                " (dataset has in=" + std::to_string(raw.input(0).size()) +
                " out=" + std::to_string(raw.target(0).size()) +
                "); dims come from the dataset, hidden layers in the middle are yours, e.g. \"2 8 8 1\"");
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
            throw std::invalid_argument("bce needs a 1-output network (use mse|cce for multi-output)");
        if (lossName == "cce" && raw.target(0).size() < 2)
            throw std::invalid_argument("cce needs one-hot multi-output targets (out dim >= 2); "
                                        "use mse|bce for 1-output networks");
        std::string optName = line("optimizer sgd|momentum|adam", "adam");
        double lr = parseDouble(line("learning rate", optName == "adam" ? "0.01" : "0.1"),
                                "learning rate");
        if (!(lr > 0.0) || lr > 5.0)
            throw std::invalid_argument("learning rate must be in (0, 5] (try 0.5 SGD/momentum, 0.01 Adam); "
                                        "got " + std::to_string(lr) + " which will diverge to NaN");
        auto opt = OptimizerFactory::create(optName, lr);
        int epochs = parseInt(line("epochs", "1500"), "epochs");
        if (epochs < 1 || epochs > 200000)
            throw std::invalid_argument("epochs must be in 1..200000");
        int batch = parseInt(line("batch size (0=full)", "8"), "batch size");
        if (batch < 0)
            throw std::invalid_argument("batch size must be >= 0 (0 = full batch)");
        if (std::size_t(batch) > trFull.size() && batch != 0) {
            std::cout << "note: batch " << batch << " > train size " << trFull.size()
                      << ", clamped to " << trFull.size() << " (full batch)\n";
            batch = int(trFull.size());
        }
        unsigned seed;
        {
            long v = parseInt(line("seed", "42"), "seed");
            if (v < 0) throw std::invalid_argument("seed must be >= 0");
            seed = static_cast<unsigned>(v);
        }

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

        HtmlReport report = HtmlReportBuilder("MiniANN playground run")
                                  .withSeries(optName + " lr=" + std::to_string(lr), h)
                                  .withBoundary(net, trFull)
                                  .withPre("network structure", ng.render())
                                  .withPre("learned weights", WeightsTable(net).render())
                                  .build();
        report.save("playground_report.html");
        std::cout << "HTML report -> playground_report.html (open in a browser)\n";

        // ---- visuals ----
        LossCurve lc(h);
        AccuracyCurve ac(h);
        WeightsTable wt(net);
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
