#include "raylib.h"

// Core MiniANN Headers
#include "miniann/network.hpp"
#include "miniann/dataset.hpp"
#include "miniann/loss.hpp"
#include "miniann/optimizer.hpp"
#include "miniann/trainer.hpp"
#include "miniann/metrics.hpp"
#include "miniann/activation.hpp"

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
#include <random>
#include <fstream>
#include <sstream>

using namespace miniann;

enum ActiveDataset { DS_AND = 0, DS_OR = 1, DS_XOR = 2, DS_IRIS = 3, DS_CSV = 4 };
enum ActiveActivation { ACT_SIGMOID = 0, ACT_TANH = 1, ACT_RELU = 2, ACT_LEAKY_RELU = 3, ACT_SWISH = 4 };
enum ActiveLoss { LOSS_MSE = 0, LOSS_BCE = 1, LOSS_CCE = 2 };
enum ActiveOptimizer { OPT_SGD = 0, OPT_MOMENTUM = 1, OPT_ADAM = 2 };
enum ActiveTab { TAB_LOSS = 0, TAB_ACCURACY = 1 };

struct LiveGuiBridge : public TrainingCallback {
    std::mutex mtx;
    std::vector<float> trainLoss;
    std::vector<float> valLoss;
    std::vector<float> trainAcc;
    std::vector<float> valAcc;
    int currentEpoch = 0;
    std::atomic<bool> isTraining{false};
    std::string evalSummary = "Ready to train.";

    void onEpoch(int epoch, const TrainingHistory& hist) override {
        std::lock_guard<std::mutex> lock(mtx);
        currentEpoch = epoch;
        if (!hist.trainLoss.empty())       trainLoss.push_back((float)hist.trainLoss.back());
        if (!hist.validationLoss.empty())  valLoss.push_back((float)hist.validationLoss.back());
        if (!hist.trainAcc.empty())        trainAcc.push_back((float)hist.trainAcc.back());
        if (!hist.validationAcc.empty())   valAcc.push_back((float)hist.validationAcc.back());
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mtx);
        trainLoss.clear();
        valLoss.clear();
        trainAcc.clear();
        valAcc.clear();
        currentEpoch = 0;
        evalSummary = "Training...";
    }
};

static LiveGuiBridge g_bridge;
static std::unique_ptr<std::thread> g_trainThread = nullptr;

static int DetectCsvColumnCount(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return 0;
    std::string line;
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        int count = 0;
        while (std::getline(ss, cell, ',')) count++;
        return count;
    }
    return 0;
}

static Dataset NormalizeMaxAbs(const Dataset& d) {
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

static Dataset OneHot(const Dataset& raw, int k) {
    Dataset d;
    for (std::size_t i = 0; i < raw.size(); ++i) {
        int c = int(raw.target(i)[0]);
        Vector t(std::size_t(k), 0.0);
        if (c >= 0 && c < k) t[std::size_t(c)] = 1.0;
        d.add(raw.input(i), t);
    }
    return d;
}

static std::string ActivationToString(ActiveActivation act) {
    if (act == ACT_SIGMOID)    return "sigmoid";
    if (act == ACT_TANH)       return "tanh";
    if (act == ACT_RELU)       return "relu";
    if (act == ACT_LEAKY_RELU) return "leaky_relu";
    return "swish";
}

void StartTrainingThread(ActiveDataset dsType, const std::string& csvPath, int targetCol,
                         const std::vector<int>& topology,
                         ActiveActivation actL1, ActiveActivation actL2,
                         ActiveLoss lossType, ActiveOptimizer optType, float lr, int totalEpochs, int batchSize) {
    if (g_trainThread && g_trainThread->joinable()) {
        g_trainThread->join();
    }

    g_bridge.reset();
    g_bridge.isTraining = true;

    g_trainThread = std::make_unique<std::thread>([=]() {
        try {
            Dataset raw;
            if (dsType == DS_AND) raw = makeAndGate();
            else if (dsType == DS_OR)  raw = makeOrGate();
            else if (dsType == DS_XOR) raw = makeXorGate();
            else if (dsType == DS_IRIS) {
                raw = Dataset::loadCSV("data/iris_small.csv", true, 4, 1);
                raw = NormalizeMaxAbs(raw);
                raw = OneHot(raw, 3);
            } else {
                int colIdx = targetCol;
                if (colIdx < 0) {
                    int totalCols = DetectCsvColumnCount(csvPath);
                    colIdx = (totalCols > 0) ? (totalCols - 1) : 0;
                }
                raw = Dataset::loadCSV(csvPath, true, colIdx, 1);
                raw = NormalizeMaxAbs(raw);
            }

            std::mt19937 srng(42);
            auto [trFull, teFull] = raw.split(0.8, srng);

            std::size_t inDim = trFull.input(0).size();
            std::size_t outDim = trFull.target(0).size();

            std::vector<int> actualSizes = topology;
            actualSizes.front() = (int)inDim;
            actualSizes.back() = (int)outDim;

            std::string actName1 = ActivationToString(actL1);
            std::string actName2 = ActivationToString(actL2);

            std::string lossName = "mse";
            if (lossType == LOSS_BCE && outDim == 1) lossName = "bce";
            else if (lossType == LOSS_CCE) lossName = "cce";

            std::string optName = "adam";
            if (optType == OPT_SGD) optName = "sgd";
            else if (optType == OPT_MOMENTUM) optName = "momentum";

            std::mt19937 rng(42);
            NeuralNetwork net;

            // Layer 1 (Hidden 1)
            net.addLayer(Layer(std::size_t(actualSizes[1]), std::size_t(actualSizes[0]),
                               ActivationFactory::create(actName1), rng, WeightInit::Xavier));

            // Layer 2 (Hidden 2)
            net.addLayer(Layer(std::size_t(actualSizes[2]), std::size_t(actualSizes[1]),
                               ActivationFactory::create(actName2), rng, WeightInit::Xavier));

            // Output Layer
            net.addLayer(Layer(std::size_t(actualSizes[3]), std::size_t(actualSizes[2]),
                               ActivationFactory::create("sigmoid"), rng, WeightInit::Xavier));

            auto loss = LossFactory::create(lossName);
            auto opt = OptimizerFactory::create(optName, (double)lr);

            TrainingConfig cfg;
            cfg.epochs = totalEpochs;
            cfg.batchSize = std::size_t(batchSize);
            cfg.shuffle = true;
            cfg.seed = 42;
            cfg.logEvery = 1;

            Trainer trainer(net, *loss, *opt, nullptr);
            TrainingHistory hist = trainer.fit(trFull, &teFull, cfg, &g_bridge);

            std::vector<Vector> pTe, tTe;
            for (std::size_t i = 0; i < teFull.size(); ++i) {
                pTe.push_back(net.predict(teFull.input(i)));
                tTe.push_back(teFull.target(i));
            }
            Accuracy accMetric;
            double testAcc = accMetric.evaluate(pTe, tTe);

            {
                std::lock_guard<std::mutex> lock(g_bridge.mtx);
                g_bridge.evalSummary = TextFormat("Final Test Acc: %.1f%%  |  Final Loss: %.4f", 
                                                  testAcc * 100.0, hist.trainLoss.empty() ? 0.0 : hist.trainLoss.back());
            }

        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(g_bridge.mtx);
            g_bridge.evalSummary = std::string("Error: ") + e.what();
        }
        g_bridge.isTraining = false;
    });
}

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(1420, 910, "MiniANN MVP - Interactive Network Workbench");
    SetWindowMinSize(1100, 720);
    SetTargetFPS(60);

    Font customFont = GetFontDefault();
    bool fontLoaded = false;
    if (FileExists("C:\\Windows\\Fonts\\segoeui.ttf")) {
        customFont = LoadFontEx("C:\\Windows\\Fonts\\segoeui.ttf", 48, nullptr, 0);
        SetTextureFilter(customFont.texture, TEXTURE_FILTER_BILINEAR);
        fontLoaded = true;
    }

    auto DrawUiText = [&](const char* text, float x, float y, float size, Color c) {
        if (fontLoaded) DrawTextEx(customFont, text, { x, y }, size, 1.0f, c);
        else DrawText(text, (int)x, (int)y, (int)size, c);
    };

    ActiveDataset datasetChoice = DS_XOR;
    std::string customCsvPath = "Drag & Drop .csv file here";
    int targetColIndex = -1;

    // Independent hidden layer activations
    ActiveActivation actChoiceL1 = ACT_TANH;
    ActiveActivation actChoiceL2 = ACT_RELU;

    ActiveLoss lossChoice = LOSS_MSE;
    ActiveOptimizer optChoice = OPT_ADAM;
    ActiveTab activeTab = TAB_LOSS;

    float learningRate = 0.05f;
    int epochsTarget = 1500;
    int batchSize = 1;

    int hidden1 = 8;
    int hidden2 = 8;

    bool isDraggingEpochSlider = false;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_F11)) ToggleFullscreen();

        if (IsFileDropped()) {
            FilePathList dropped = LoadDroppedFiles();
            if (dropped.count > 0) {
                std::string path = dropped.paths[0];
                if (path.length() >= 4 && path.substr(path.length() - 4) == ".csv") {
                    customCsvPath = path;
                    datasetChoice = DS_CSV;
                    int cols = DetectCsvColumnCount(path);
                    targetColIndex = (cols > 0) ? (cols - 1) : 0;
                }
            }
            UnloadDroppedFiles(dropped);
        }

        float sw = (float)GetScreenWidth();
        float sh = (float)GetScreenHeight();
        float s = std::clamp(sh / 880.0f, 0.82f, 1.6f);
        Vector2 mouse = GetMousePosition();

        float topH = 68 * s;
        float bottomH = 46 * s;
        float mainY = topH + 14 * s;
        float mainH = sh - mainY - bottomH - 14 * s;

        Rectangle leftCard = { 35 * s, mainY, 450 * s, mainH };
        Rectangle rightCard = { leftCard.x + leftCard.width + 25 * s, mainY, sw - (leftCard.x + leftCard.width + 60 * s), mainH };

        int inDimDisplay = (datasetChoice == DS_IRIS) ? 4 : (datasetChoice == DS_CSV) ? 4 : 2;
        int outDimDisplay = (datasetChoice == DS_IRIS) ? 3 : 1;

        float btnH = 32 * s;
        float dsW = (leftCard.width - 48 * s) / 5.0f;
        Rectangle btnDS0 = { leftCard.x + 20 * s, leftCard.y + 45 * s, dsW, btnH };
        Rectangle btnDS1 = { btnDS0.x + dsW + 3 * s, btnDS0.y, dsW, btnH };
        Rectangle btnDS2 = { btnDS1.x + dsW + 3 * s, btnDS0.y, dsW, btnH };
        Rectangle btnDS3 = { btnDS2.x + dsW + 3 * s, btnDS0.y, dsW, btnH };
        Rectangle btnDS4 = { btnDS3.x + dsW + 3 * s, btnDS0.y, dsW, btnH };

        // Architecture Step Buttons
        float archY = btnDS0.y + btnH + 42 * s;
        Rectangle btnH1Minus = { leftCard.x + 130 * s, archY + 24 * s, 30 * s, 28 * s };
        Rectangle btnH1Plus  = { leftCard.x + 165 * s, archY + 24 * s, 30 * s, 28 * s };
        Rectangle btnH2Minus = { leftCard.x + 315 * s, archY + 24 * s, 30 * s, 28 * s };
        Rectangle btnH2Plus  = { leftCard.x + 350 * s, archY + 24 * s, 30 * s, 28 * s };

        // Epoch Slider Layout
        float epochY = archY + 66 * s;
        Rectangle epochTrack = { leftCard.x + 20 * s, epochY + 28 * s, leftCard.width - 150 * s, 10 * s };
        float epochNorm = std::clamp((float)(epochsTarget - 100) / 4900.0f, 0.0f, 1.0f);
        Rectangle epochHandle = { epochTrack.x + epochNorm * epochTrack.width - 6 * s, epochTrack.y - 6 * s, 12 * s, 22 * s };
        Rectangle btnEpochMinus = { leftCard.x + leftCard.width - 110 * s, epochY + 18 * s, 42 * s, 26 * s };
        Rectangle btnEpochPlus  = { leftCard.x + leftCard.width - 60 * s,  epochY + 18 * s, 42 * s, 26 * s };

        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && 
            (CheckCollisionPointRec(mouse, epochTrack) || CheckCollisionPointRec(mouse, epochHandle))) {
            isDraggingEpochSlider = true;
        }
        if (isDraggingEpochSlider) {
            if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
                float norm = std::clamp((mouse.x - epochTrack.x) / epochTrack.width, 0.0f, 1.0f);
                epochsTarget = 100 + (int)(norm * 4900.0f);
                epochsTarget = (epochsTarget / 50) * 50;
            } else {
                isDraggingEpochSlider = false;
            }
        }

        // --- Per-Layer Activation Selectors (5 buttons per layer) ---
        float actW = (leftCard.width - 48 * s) / 5.0f;
        float actY_L1 = epochY + 58 * s;
        Rectangle btnL1_0 = { leftCard.x + 20 * s, actY_L1 + 22 * s, actW, 28 * s };
        Rectangle btnL1_1 = { btnL1_0.x + actW + 2 * s, btnL1_0.y, actW, 28 * s };
        Rectangle btnL1_2 = { btnL1_1.x + actW + 2 * s, btnL1_0.y, actW, 28 * s };
        Rectangle btnL1_3 = { btnL1_2.x + actW + 2 * s, btnL1_0.y, actW, 28 * s };
        Rectangle btnL1_4 = { btnL1_3.x + actW + 2 * s, btnL1_0.y, actW, 28 * s };

        float actY_L2 = actY_L1 + 62 * s;
        Rectangle btnL2_0 = { leftCard.x + 20 * s, actY_L2 + 22 * s, actW, 28 * s };
        Rectangle btnL2_1 = { btnL2_0.x + actW + 2 * s, btnL2_0.y, actW, 28 * s };
        Rectangle btnL2_2 = { btnL2_1.x + actW + 2 * s, btnL2_0.y, actW, 28 * s };
        Rectangle btnL2_3 = { btnL2_2.x + actW + 2 * s, btnL2_0.y, actW, 28 * s };
        Rectangle btnL2_4 = { btnL2_3.x + actW + 2 * s, btnL2_0.y, actW, 28 * s };

        // Loss & Optimizer
        float lossW = (leftCard.width - 46 * s) / 3.0f;
        float lossStartY = actY_L2 + 62 * s;
        Rectangle btnLoss0 = { leftCard.x + 20 * s, lossStartY + 22 * s, lossW, btnH };
        Rectangle btnLoss1 = { btnLoss0.x + lossW + 5 * s, btnLoss0.y, lossW, btnH };
        Rectangle btnLoss2 = { btnLoss1.x + lossW + 5 * s, btnLoss0.y, lossW, btnH };

        Rectangle btnOpt0 = { leftCard.x + 20 * s, btnLoss0.y + btnH + 28 * s, lossW, btnH };
        Rectangle btnOpt1 = { btnOpt0.x + lossW + 5 * s, btnOpt0.y, lossW, btnH };
        Rectangle btnOpt2 = { btnOpt1.x + lossW + 5 * s, btnOpt0.y, lossW, btnH };

        Rectangle btnTrain = { leftCard.x + 20 * s, leftCard.y + leftCard.height - 56 * s, leftCard.width - 40 * s, 46 * s };

        Rectangle tabLoss = { rightCard.x + 25 * s, rightCard.y + 15 * s, 140 * s, 34 * s };
        Rectangle tabAcc  = { tabLoss.x + tabLoss.width + 10 * s, tabLoss.y, 150 * s, 34 * s };

        // Handle Clicks
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && !isDraggingEpochSlider) {
            if (CheckCollisionPointRec(mouse, btnDS0)) datasetChoice = DS_AND;
            if (CheckCollisionPointRec(mouse, btnDS1)) datasetChoice = DS_OR;
            if (CheckCollisionPointRec(mouse, btnDS2)) datasetChoice = DS_XOR;
            if (CheckCollisionPointRec(mouse, btnDS3)) datasetChoice = DS_IRIS;
            if (CheckCollisionPointRec(mouse, btnDS4)) datasetChoice = DS_CSV;

            if (CheckCollisionPointRec(mouse, btnH1Minus)) hidden1 = std::max(1, hidden1 - 1);
            if (CheckCollisionPointRec(mouse, btnH1Plus))  hidden1 = std::min(64, hidden1 + 1);
            if (CheckCollisionPointRec(mouse, btnH2Minus)) hidden2 = std::max(1, hidden2 - 1);
            if (CheckCollisionPointRec(mouse, btnH2Plus))  hidden2 = std::min(64, hidden2 + 1);

            if (CheckCollisionPointRec(mouse, btnEpochMinus)) epochsTarget = std::max(100, epochsTarget - 100);
            if (CheckCollisionPointRec(mouse, btnEpochPlus))  epochsTarget = std::min(10000, epochsTarget + 100);

            // Layer 1 activation selection
            if (CheckCollisionPointRec(mouse, btnL1_0)) actChoiceL1 = ACT_SIGMOID;
            if (CheckCollisionPointRec(mouse, btnL1_1)) actChoiceL1 = ACT_TANH;
            if (CheckCollisionPointRec(mouse, btnL1_2)) actChoiceL1 = ACT_RELU;
            if (CheckCollisionPointRec(mouse, btnL1_3)) actChoiceL1 = ACT_LEAKY_RELU;
            if (CheckCollisionPointRec(mouse, btnL1_4)) actChoiceL1 = ACT_SWISH;

            // Layer 2 activation selection
            if (CheckCollisionPointRec(mouse, btnL2_0)) actChoiceL2 = ACT_SIGMOID;
            if (CheckCollisionPointRec(mouse, btnL2_1)) actChoiceL2 = ACT_TANH;
            if (CheckCollisionPointRec(mouse, btnL2_2)) actChoiceL2 = ACT_RELU;
            if (CheckCollisionPointRec(mouse, btnL2_3)) actChoiceL2 = ACT_LEAKY_RELU;
            if (CheckCollisionPointRec(mouse, btnL2_4)) actChoiceL2 = ACT_SWISH;

            if (CheckCollisionPointRec(mouse, btnLoss0)) lossChoice = LOSS_MSE;
            if (CheckCollisionPointRec(mouse, btnLoss1)) lossChoice = LOSS_BCE;
            if (CheckCollisionPointRec(mouse, btnLoss2)) lossChoice = LOSS_CCE;

            if (CheckCollisionPointRec(mouse, btnOpt0)) optChoice = OPT_SGD;
            if (CheckCollisionPointRec(mouse, btnOpt1)) optChoice = OPT_MOMENTUM;
            if (CheckCollisionPointRec(mouse, btnOpt2)) optChoice = OPT_ADAM;

            if (CheckCollisionPointRec(mouse, tabLoss)) activeTab = TAB_LOSS;
            if (CheckCollisionPointRec(mouse, tabAcc))  activeTab = TAB_ACCURACY;

            if (CheckCollisionPointRec(mouse, btnTrain)) {
                std::vector<int> topology = { inDimDisplay, hidden1, hidden2, outDimDisplay };
                StartTrainingThread(datasetChoice, customCsvPath, targetColIndex, topology, 
                                    actChoiceL1, actChoiceL2, lossChoice, optChoice, learningRate, epochsTarget, batchSize);
            }
        }

        // --- DRAW UI ---
        BeginDrawing();
        ClearBackground(GetColor(0x0C0D14FF));

        // Top Navigation
        DrawRectangle(0, 0, (int)sw, (int)topH, GetColor(0x151722FF));
        DrawLine(0, (int)topH, (int)sw, (int)topH, GetColor(0x232738FF));
        DrawUiText("MiniANN MVP - Network Workbench", 35 * s, 18 * s, 26 * s, RAYWHITE);

        std::vector<float> trainLossCopy, valLossCopy, trainAccCopy, valAccCopy;
        int liveEpoch = 0;
        bool isTrainingLive = g_bridge.isTraining;
        std::string summaryText;
        {
            std::lock_guard<std::mutex> lock(g_bridge.mtx);
            trainLossCopy = g_bridge.trainLoss;
            valLossCopy = g_bridge.valLoss;
            trainAccCopy = g_bridge.trainAcc;
            valAccCopy = g_bridge.valAcc;
            liveEpoch = g_bridge.currentEpoch;
            summaryText = g_bridge.evalSummary;
        }

        float curLossVal = trainLossCopy.empty() ? 0.0f : trainLossCopy.back();
        float curAccVal  = trainAccCopy.empty()  ? 0.0f : trainAccCopy.back();

        DrawUiText(TextFormat("Epoch: %d / %d", liveEpoch, epochsTarget), sw - 560 * s, 22 * s, 17 * s, LIGHTGRAY);
        DrawUiText(TextFormat("Loss: %.4f", curLossVal), sw - 390 * s, 22 * s, 17 * s, YELLOW);
        DrawUiText(TextFormat("Acc: %.1f%%", curAccVal * 100.0f), sw - 220 * s, 22 * s, 17 * s, GREEN);

        Rectangle statusBadge = { sw - 95 * s, 18 * s, 70 * s, 32 * s };
        DrawRectangleRounded(statusBadge, 0.4f, 4, isTrainingLive ? GetColor(0x14532DFF) : GetColor(0x27272AFF));
        DrawUiText(isTrainingLive ? "RUN" : "IDLE", statusBadge.x + 16 * s, statusBadge.y + 7 * s, 14 * s, isTrainingLive ? GREEN : LIGHTGRAY);

        // ==========================================
        // LEFT CARD: CONFIGURATION
        // ==========================================
        DrawRectangleRec(leftCard, GetColor(0x13151FFF));
        DrawRectangleLinesEx(leftCard, 1.5f, GetColor(0x25293BFF));

        auto drawBtn = [&](Rectangle r, const char* label, bool active, float fontSize = 13.0f) {
            DrawRectangleRec(r, active ? GetColor(0x2563EBFF) : GetColor(0x1B1E2BFF));
            DrawRectangleLinesEx(r, 1.0f, active ? SKYBLUE : GetColor(0x2E3448FF));
            float tw = (float)MeasureText(label, (int)(fontSize * s));
            DrawUiText(label, r.x + (r.width - tw) / 2, r.y + (r.height - fontSize * s) / 2, fontSize * s, RAYWHITE);
        };

        // 1. Dataset
        DrawUiText("1. DATASET", leftCard.x + 20 * s, leftCard.y + 18 * s, 15 * s, RAYWHITE);
        drawBtn(btnDS0, "AND", datasetChoice == DS_AND);
        drawBtn(btnDS1, "OR",  datasetChoice == DS_OR);
        drawBtn(btnDS2, "XOR", datasetChoice == DS_XOR);
        drawBtn(btnDS3, "Iris", datasetChoice == DS_IRIS);
        drawBtn(btnDS4, "CSV", datasetChoice == DS_CSV);

        // 2. NN Architecture Sizing
        DrawUiText("2. ARCHITECTURE SIZES", leftCard.x + 20 * s, archY, 15 * s, RAYWHITE);
        std::string archStr = TextFormat("[%d -> %d -> %d -> %d]", inDimDisplay, hidden1, hidden2, outDimDisplay);
        DrawUiText(archStr.c_str(), leftCard.x + 210 * s, archY, 15 * s, SKYBLUE);

        DrawUiText(TextFormat("H1: %d", hidden1), leftCard.x + 20 * s, archY + 28 * s, 14 * s, LIGHTGRAY);
        drawBtn(btnH1Minus, "-", false);
        drawBtn(btnH1Plus,  "+", false);

        DrawUiText(TextFormat("H2: %d", hidden2), leftCard.x + 210 * s, archY + 28 * s, 14 * s, LIGHTGRAY);
        drawBtn(btnH2Minus, "-", false);
        drawBtn(btnH2Plus,  "+", false);

        // 3. Epoch Length Slider
        DrawUiText("3. TOTAL EPOCHS", leftCard.x + 20 * s, epochY, 15 * s, RAYWHITE);
        DrawUiText(TextFormat("%d", epochsTarget), leftCard.x + 150 * s, epochY, 15 * s, YELLOW);

        DrawRectangleRec(epochTrack, GetColor(0x2A2E40FF));
        DrawRectangle((int)epochTrack.x, (int)epochTrack.y, (int)(epochNorm * epochTrack.width), (int)epochTrack.height, SKYBLUE);
        DrawRectangleRec(epochHandle, RAYWHITE);
        drawBtn(btnEpochMinus, "-100", false, 11.0f);
        drawBtn(btnEpochPlus,  "+100", false, 11.0f);

        // 4. Hidden Layer 1 Activation
        DrawUiText("4. HIDDEN 1 ACTIVATION", leftCard.x + 20 * s, actY_L1, 14 * s, RAYWHITE);
        drawBtn(btnL1_0, "Sig",   actChoiceL1 == ACT_SIGMOID, 11.0f);
        drawBtn(btnL1_1, "Tanh",  actChoiceL1 == ACT_TANH, 11.0f);
        drawBtn(btnL1_2, "ReLU",  actChoiceL1 == ACT_RELU, 11.0f);
        drawBtn(btnL1_3, "L-ReLU", actChoiceL1 == ACT_LEAKY_RELU, 10.0f);
        drawBtn(btnL1_4, "Swish", actChoiceL1 == ACT_SWISH, 11.0f);

        // 5. Hidden Layer 2 Activation
        DrawUiText("5. HIDDEN 2 ACTIVATION", leftCard.x + 20 * s, actY_L2, 14 * s, RAYWHITE);
        drawBtn(btnL2_0, "Sig",   actChoiceL2 == ACT_SIGMOID, 11.0f);
        drawBtn(btnL2_1, "Tanh",  actChoiceL2 == ACT_TANH, 11.0f);
        drawBtn(btnL2_2, "ReLU",  actChoiceL2 == ACT_RELU, 11.0f);
        drawBtn(btnL2_3, "L-ReLU", actChoiceL2 == ACT_LEAKY_RELU, 10.0f);
        drawBtn(btnL2_4, "Swish", actChoiceL2 == ACT_SWISH, 11.0f);

        // 6. Loss & Optimizer
        DrawUiText("6. LOSS & OPTIMIZER", leftCard.x + 20 * s, lossStartY, 14 * s, RAYWHITE);
        drawBtn(btnLoss0, "MSE", lossChoice == LOSS_MSE);
        drawBtn(btnLoss1, "BCE", lossChoice == LOSS_BCE);
        drawBtn(btnLoss2, "CCE", lossChoice == LOSS_CCE);

        drawBtn(btnOpt0, "SGD",  optChoice == OPT_SGD);
        drawBtn(btnOpt1, "Mom.", optChoice == OPT_MOMENTUM);
        drawBtn(btnOpt2, "Adam", optChoice == OPT_ADAM);

        // Start Training Button
        DrawRectangleRec(btnTrain, isTrainingLive ? GetColor(0x7F1D1DFF) : GetColor(0x16A34AFF));
        DrawRectangleLinesEx(btnTrain, 1.5f, isTrainingLive ? RED : GREEN);
        const char* trainLabel = isTrainingLive ? "TRAINING IN PROGRESS..." : "START TRAINING";
        float trainTw = (float)MeasureText(trainLabel, (int)(16 * s));
        DrawUiText(trainLabel, btnTrain.x + (btnTrain.width - trainTw) / 2, btnTrain.y + (btnTrain.height - 16 * s) / 2, 16 * s, RAYWHITE);

        // ==========================================
        // RIGHT CARD: PLOTTER
        // ==========================================
        DrawRectangleRec(rightCard, GetColor(0x13151FFF));
        DrawRectangleLinesEx(rightCard, 1.5f, GetColor(0x25293BFF));

        drawBtn(tabLoss, "Loss Curves", activeTab == TAB_LOSS);
        drawBtn(tabAcc,  "Accuracy (%)", activeTab == TAB_ACCURACY);

        Rectangle graphArea = { rightCard.x + 50 * s, rightCard.y + 65 * s, rightCard.width - 80 * s, rightCard.height - 130 * s };
        DrawRectangleRec(graphArea, GetColor(0x090A0FFF));
        DrawRectangleLinesEx(graphArea, 1.5f, GetColor(0x2E3448FF));

        const std::vector<float>& mainData = (activeTab == TAB_LOSS) ? trainLossCopy : trainAccCopy;
        const std::vector<float>& valData  = (activeTab == TAB_LOSS) ? valLossCopy   : valAccCopy;

        float maxVal = (activeTab == TAB_ACCURACY) ? 1.0f : 1.0f;
        for (float v : mainData) if (v > maxVal) maxVal = v;
        for (float v : valData)  if (v > maxVal) maxVal = v;
        if (activeTab == TAB_LOSS) maxVal = std::ceil(maxVal * 10.0f) / 10.0f;

        for (int i = 0; i <= 5; ++i) {
            float norm = (float)i / 5.0f;
            float gy = graphArea.y + graphArea.height - (norm * graphArea.height);
            DrawLine((int)graphArea.x, (int)gy, (int)(graphArea.x + graphArea.width), (int)gy, GetColor(0x181B26FF));

            float yVal = norm * maxVal;
            DrawUiText(TextFormat((activeTab == TAB_ACCURACY) ? "%.0f%%" : "%.2f", (activeTab == TAB_ACCURACY) ? yVal * 100.0f : yVal),
                       graphArea.x - 48 * s, gy - 7 * s, 12 * s, GRAY);
        }

        DrawUiText("0", graphArea.x, graphArea.y + graphArea.height + 8 * s, 13 * s, GRAY);
        DrawUiText(TextFormat("%d", liveEpoch), graphArea.x + graphArea.width - 35 * s, graphArea.y + graphArea.height + 8 * s, 13 * s, GRAY);
        DrawUiText("Epochs ->", graphArea.x + graphArea.width / 2 - 30 * s, graphArea.y + graphArea.height + 8 * s, 13 * s, LIGHTGRAY);

        if (mainData.size() > 1) {
            for (size_t i = 1; i < mainData.size(); ++i) {
                float x1 = graphArea.x + ((float)(i - 1) / (mainData.size() - 1)) * graphArea.width;
                float y1 = graphArea.y + graphArea.height - (mainData[i - 1] / maxVal) * graphArea.height;
                float x2 = graphArea.x + ((float)i / (mainData.size() - 1)) * graphArea.width;
                float y2 = graphArea.y + graphArea.height - (mainData[i] / maxVal) * graphArea.height;
                DrawLineEx({x1, y1}, {x2, y2}, 2.5f * s, (activeTab == TAB_LOSS) ? SKYBLUE : GREEN);
            }
        }

        if (valData.size() > 1) {
            for (size_t i = 1; i < valData.size(); ++i) {
                float x1 = graphArea.x + ((float)(i - 1) / (valData.size() - 1)) * graphArea.width;
                float y1 = graphArea.y + graphArea.height - (valData[i - 1] / maxVal) * graphArea.height;
                float x2 = graphArea.x + ((float)i / (valData.size() - 1)) * graphArea.width;
                float y2 = graphArea.y + graphArea.height - (valData[i] / maxVal) * graphArea.height;
                DrawLineEx({x1, y1}, {x2, y2}, 2.0f * s, ORANGE);
            }
        }

        DrawCircle((int)(graphArea.x + graphArea.width - 240 * s), (int)(graphArea.y + 20 * s), 5 * s, (activeTab == TAB_LOSS) ? SKYBLUE : GREEN);
        DrawUiText("Train Set", graphArea.x + graphArea.width - 225 * s, graphArea.y + 12 * s, 14 * s, LIGHTGRAY);
        DrawCircle((int)(graphArea.x + graphArea.width - 120 * s), (int)(graphArea.y + 20 * s), 5 * s, ORANGE);
        DrawUiText("Validation", graphArea.x + graphArea.width - 105 * s, graphArea.y + 12 * s, 14 * s, LIGHTGRAY);

        Rectangle summaryBar = { rightCard.x + 50 * s, graphArea.y + graphArea.height + 34 * s, graphArea.width, 34 * s };
        DrawRectangleRec(summaryBar, GetColor(0x181A24FF));
        DrawRectangleLinesEx(summaryBar, 1.0f, GetColor(0x2A2E40FF));
        DrawUiText(summaryText.c_str(), summaryBar.x + 15 * s, summaryBar.y + 8 * s, 13 * s, YELLOW);

        if (CheckCollisionPointRec(mouse, graphArea) && mainData.size() > 1) {
            DrawLine((int)mouse.x, (int)graphArea.y, (int)mouse.x, (int)(graphArea.y + graphArea.height), ColorAlpha(WHITE, 0.25f));
            float relX = (mouse.x - graphArea.x) / graphArea.width;
            int idx = std::clamp((int)(relX * (mainData.size() - 1)), 0, (int)mainData.size() - 1);

            Rectangle tip = { mouse.x + 12 * s, mouse.y - 50 * s, 140 * s, 48 * s };
            DrawRectangleRounded(tip, 0.2f, 4, GetColor(0x000000EE));
            DrawRectangleLinesEx(tip, 1.0f, DARKGRAY);

            DrawUiText(TextFormat("Epoch: %d", idx + 1), tip.x + 8 * s, tip.y + 6 * s, 12 * s, WHITE);
            DrawUiText(TextFormat("Train: %.4f", mainData[idx]), tip.x + 8 * s, tip.y + 20 * s, 12 * s, (activeTab == TAB_LOSS) ? SKYBLUE : GREEN);
            if (idx < (int)valData.size()) {
                DrawUiText(TextFormat("Val  : %.4f", valData[idx]), tip.x + 72 * s, tip.y + 20 * s, 12 * s, ORANGE);
            }
        }

        // Footer
        DrawRectangle(0, (int)(sh - bottomH), (int)sw, (int)bottomH, GetColor(0x11131CFF));
        DrawLine(0, (int)(sh - bottomH), (int)sw, (int)(sh - bottomH), GetColor(0x232738FF));
        DrawUiText("[F11] Fullscreen  |  Pick different activations for H1 & H2, adjust epochs, and hit START TRAINING",
                   sw / 2 - 380 * s, sh - bottomH + 14 * s, 14 * s, GRAY);

        EndDrawing();
    }

    if (g_trainThread && g_trainThread->joinable()) {
        g_trainThread->join();
    }
    if (fontLoaded) UnloadFont(customFont);
    CloseWindow();
    return 0;
}