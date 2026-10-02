#pragma once
// Header-only CLI: lets the user choose descent method, training setup,
// and activations without touching code. All demos share it.
#include "miniann/activation.hpp"
#include "miniann/optimizer.hpp"
#include "miniann/trainer.hpp"
#include <iostream>
#include <memory>
#include <string>

namespace miniann {
namespace cli {

struct Options {
    std::string optimizer = "sgd";   // sgd | momentum | adam
    double lr = 0.5;                 // learning rate
    std::string hiddenAct = "tanh";  // sigmoid | tanh | relu | leaky_relu | swish
    int epochs = 10000;
    int batch = 1;                   // 1 = online, >1 = mini-batch, 0 = full batch
    unsigned seed = 42;
    int hidden = 4;
    bool live = true;                // live redraw while training (ANSI terminal)
};

// He init suits ReLU-family (relu/leaky_relu), Xavier suits sigmoid/tanh/swish.
inline WeightInit defaultInitFor(const std::string& act) {
    if (act == "relu" || act == "leaky_relu" || act == "leakyrelu") return WeightInit::He;
    return WeightInit::Xavier;
}

inline void printUsage(const char* prog) {
    std::cout << "usage: " << prog << " [options]\n"
              << "  --opt sgd|momentum|adam   descent method (default sgd)\n"
              << "  --lr <rate>               learning rate (default 0.5 for sgd, use 0.01 for adam)\n"
              << "  --act <name>              hidden activation: sigmoid|tanh|relu|leaky_relu|swish\n"
              << "  --epochs <n>  --batch <n> --seed <n>  --hidden <n>\n"
              << "  --live | --no-live      redraw plots + neurons live (default on)\n";
}

// Parses argv; unknown flags print usage and throw invalid_argument.
inline Options parse(int argc, char** argv, Options base = Options{}) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* flag) -> std::string {
            if (++i >= argc) throw std::invalid_argument(std::string("missing value for ") + flag);
            return argv[i];
        };
        if (a == "--opt") base.optimizer = need("--opt");
        else if (a == "--lr") base.lr = std::stod(need("--lr"));
        else if (a == "--act") base.hiddenAct = need("--act");
        else if (a == "--epochs") base.epochs = std::stoi(need("--epochs"));
        else if (a == "--batch") base.batch = std::stoi(need("--batch"));
        else if (a == "--seed") base.seed = unsigned(std::stoul(need("--seed")));
        else if (a == "--hidden") base.hidden = std::stoi(need("--hidden"));
        else if (a == "--live") base.live = true;
        else if (a == "--no-live") base.live = false;
        else if (a == "--help" || a == "-h") { printUsage(argv[0]); std::exit(0); }
        else { printUsage(argv[0]); throw std::invalid_argument("unknown flag '" + a + "'"); }
    }
    // Validate through the real factories so typos fail fast with a good message.
    ActivationFactory::create(base.hiddenAct);
    OptimizerFactory::create(base.optimizer, base.lr);
    return base;
}

inline std::unique_ptr<IOptimizer> makeOptimizer(const Options& o) {
    return OptimizerFactory::create(o.optimizer, o.lr);
}

} // namespace cli
} // namespace miniann
