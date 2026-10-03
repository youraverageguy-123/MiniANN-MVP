#pragma once
// Shared GUI contract: the Qt workbench (demos/gui_qt.cpp, friend's) is a
// thin view over the SAME core library. It never owns training math.
// Friend works in gui_qt/experiment files, I work in
// cli/report/playground/library — the only shared surface is this contract
// + the model file schema.
#include <string>
#include <vector>

namespace miniann {
namespace gui {

// What the workbench supports. Keeps the UI feature set verifiable
// in viva against a single checklist.
struct WorkbenchConfig {
    std::string dataset = "xor"; // and|or|xor|iris|csv
    std::string csvPath;
    int csvTargetCol = -1; // -1 = last col
    std::vector<int> topology = {2, 8, 8, 1};
    std::string hiddenAct1 = "tanh";
    std::string hiddenAct2 = "relu";
    std::string outputAct = "sigmoid"; // sigmoid|softmax|linear
    std::string loss = "mse";          // mse|bce|cce
    std::string optimizer = "adam";    // sgd|momentum|adam
    double lr = 0.05;
    int epochs = 1500;
    int batchSize = 1;
    unsigned seed = 42;
    double threshold = 0.5; // binary decision threshold (wires to Accuracy/CM)
};

// File layout so ownership never collides:
//   friend (Qt GUI): demos/gui_qt.cpp, include/miniann/experiment.hpp,
//                    src/experiment.cpp, tests/test_training.cpp
//   shared (both):  include/miniann/{network,dataset,trainer,metrics,
//                   serializer,visualizer,report,cli,gui_contract}.hpp
// Rule: GUIs may only call public core APIs + cli::parse/validate +
// ModelSerializer::save/loadWithOptimizer. No GUI writes into src/.

} // namespace gui
} // namespace miniann
