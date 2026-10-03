#pragma once
// Header-only CLI: lets the user choose descent method, training setup,
// and activations without touching code. All demos share it.
#include "miniann/activation.hpp"
#include "miniann/optimizer.hpp"
#include "miniann/trainer.hpp"
#include <iostream>
#include <memory>
#include <sstream>
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
    std::string report;              // "" = demo default (e.g. xor_report.html)
    bool noReport = false;           // --no-report: skip HTML report entirely
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
              << "  --live | --no-live      redraw plots + neurons live (default on)\n"
              << "  --report <path>         HTML report path (default per demo)\n"
              << "  --no-report             skip writing the HTML report\n";
}

// Validate ranges + factory tokens so typos fail fast with a good message.
inline void validate(const Options& o) {
    ActivationFactory::create(o.hiddenAct);
    OptimizerFactory::create(o.optimizer, o.lr);
    if (!(o.lr > 0.0) || o.lr > 5.0)
        throw std::invalid_argument("learning rate must be in (0, 5]");
    if (o.epochs < 1 || o.epochs > 200000)
        throw std::invalid_argument("epochs must be in 1..200000");
    if (o.batch < 0)
        throw std::invalid_argument("batch must be >= 0 (0 = full batch)");
    if (o.hidden < 1)
        throw std::invalid_argument("hidden must be >= 1");
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
        else if (a == "--report") base.report = need("--report");
        else if (a == "--no-report") base.noReport = true;
        else if (a == "--help" || a == "-h") { printUsage(argv[0]); std::exit(0); }
        else { printUsage(argv[0]); throw std::invalid_argument("unknown flag '" + a + "'"); }
    }
    // Validate through the real factories so typos fail fast with a good message.
    validate(base);
    return base;
}

inline std::unique_ptr<IOptimizer> makeOptimizer(const Options& o) {
    return OptimizerFactory::create(o.optimizer, o.lr);
}

// Reconstruct the exact command line for reproducibility.
inline std::string commandLine(int argc, char** argv) {
    std::string s;
    for (int i = 0; i < argc; ++i) {
        if (i) s += " ";
        std::string a = argv[i];
        if (a.find(' ') != std::string::npos) s += "\"" + a + "\"";
        else s += a;
    }
    return s;
}

// Human-readable dump of every CLI knob. Stable order, one line per knob,
// so it renders well inside HtmlReport <pre> cards and console logs.
inline std::string describe(const Options& o, const std::string& cmdLine,
                            const std::string& defaultReport = "") {
    std::ostringstream os;
    os << "command: " << cmdLine << "\n";
    os << "optimizer : " << o.optimizer << "\n";
    os << "lr        : " << o.lr << "\n";
    os << "hidden act: " << o.hiddenAct << " (init "
       << (defaultInitFor(o.hiddenAct) == WeightInit::He ? "He" : "Xavier") << ")\n";
    os << "epochs    : " << o.epochs << "\n";
    os << "batch     : " << o.batch
       << (o.batch == 0 ? " (full batch)" : o.batch == 1 ? " (online SGD)" : " (mini-batch)") << "\n";
    os << "seed      : " << o.seed << "\n";
    os << "hidden    : " << o.hidden << "\n";
    os << "live      : " << (o.live ? "on" : "off") << "\n";
    os << "report    : " << (o.noReport ? "(skipped via --no-report)"
                                        : (o.report.empty() ? defaultReport : o.report)) << "\n";
    return os.str();
}

// Resolve where the demo should write its HTML report.
// Returns "" when --no-report was given (caller should skip saving).
inline std::string resolveReportPath(const Options& o, const std::string& demoDefault) {
    if (o.noReport) return "";
    return o.report.empty() ? demoDefault : o.report;
}

// Backward-compatible overload: builds the command line from argc/argv.
inline std::string describe(const Options& o, int argc, char** argv,
                            const std::string& defaultReport = "") {
    return describe(o, commandLine(argc, argv), defaultReport);
}

} // namespace cli
} // namespace miniann
