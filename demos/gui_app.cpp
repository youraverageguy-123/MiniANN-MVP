#include "raylib.h"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

enum DatasetType {
    DATASET_XOR = 0,
    DATASET_CIRCLES = 1,
    DATASET_AND = 2
};

enum ActivationType {
    ACT_RELU = 0,
    ACT_LEAKY_RELU = 1,
    ACT_SIGMOID = 2,
    ACT_TANH = 3,
    ACT_SWISH = 4
};

enum GraphTab {
    TAB_LOSS = 0,
    TAB_ACCURACY = 1,
    TAB_GRAD_NORM = 2
};

struct DataPoint {
    float x, y;
    int label;
};

std::vector<DataPoint> LoadDataset(DatasetType type) {
    if (type == DATASET_XOR) {
        return {
            {0.15f, 0.15f, 0}, {0.20f, 0.22f, 0},
            {0.80f, 0.85f, 0}, {0.85f, 0.80f, 0},
            {0.18f, 0.82f, 1}, {0.25f, 0.78f, 1},
            {0.82f, 0.18f, 1}, {0.78f, 0.24f, 1}
        };
    } else if (type == DATASET_CIRCLES) {
        return {
            {0.50f, 0.50f, 1}, {0.45f, 0.55f, 1}, {0.55f, 0.45f, 1},
            {0.15f, 0.50f, 0}, {0.85f, 0.50f, 0}, {0.50f, 0.15f, 0}, {0.50f, 0.85f, 0}
        };
    } else {
        return {
            {0.20f, 0.20f, 0}, {0.20f, 0.80f, 0}, {0.80f, 0.20f, 0},
            {0.80f, 0.80f, 1}, {0.85f, 0.75f, 1}
        };
    }
}

int main() {
    // Enable resizable and high-DPI flags before InitWindow
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(1440, 840, "MiniANN MVP - Interactive Network Inspector");
    SetWindowMinSize(1100, 680);
    SetTargetFPS(60);

    // Load large crisp font (48px base for clean high-res scaling)
    Font customFont = GetFontDefault();
    bool customFontLoaded = false;
    if (FileExists("C:\\Windows\\Fonts\\segoeui.ttf")) {
        customFont = LoadFontEx("C:\\Windows\\Fonts\\segoeui.ttf", 48, nullptr, 0);
        SetTextureFilter(customFont.texture, TEXTURE_FILTER_BILINEAR);
        customFontLoaded = true;
    } else if (FileExists("C:\\Windows\\Fonts\\arial.ttf")) {
        customFont = LoadFontEx("C:\\Windows\\Fonts\\arial.ttf", 48, nullptr, 0);
        SetTextureFilter(customFont.texture, TEXTURE_FILTER_BILINEAR);
        customFontLoaded = true;
    }

    auto DrawCustomText = [&](const char* text, float posX, float posY, float fontSize, Color color) {
        if (customFontLoaded) {
            DrawTextEx(customFont, text, { posX, posY }, fontSize, 1.0f, color);
        } else {
            DrawText(text, (int)posX, (int)posY, (int)fontSize, color);
        }
    };

    DatasetType currentDatasetType = DATASET_XOR;
    ActivationType currentActivation = ACT_RELU;
    GraphTab activeTab = TAB_LOSS;
    std::vector<DataPoint> dataset = LoadDataset(currentDatasetType);

    std::vector<float> lossHistory;
    std::vector<float> accHistory;
    std::vector<float> gradHistory;

    bool isPaused = false;
    int epoch = 0;
    float currentLoss = 0.95f;
    float currentAcc = 50.0f;
    float currentGrad = 0.5f;
    float learningRate = 0.10f;

    float simSpeed = 0.6f; 
    bool isDraggingSlider = false;
    float stepAccumulator = 0.0f;

    const int gridRes = 90;
    RenderTexture2D boundaryTex = LoadRenderTexture(gridRes, gridRes);

    while (!WindowShouldClose()) {
        // --- 1. FULLSCREEN & KEYBOARD CONTROLS ---
        if (IsKeyPressed(KEY_F11)) {
            ToggleFullscreen();
        }
        if (IsKeyPressed(KEY_SPACE)) isPaused = !isPaused;
        if (IsKeyPressed(KEY_R)) {
            lossHistory.clear(); accHistory.clear(); gradHistory.clear();
            epoch = 0;
            currentLoss = 0.95f; currentAcc = 50.0f; currentGrad = 0.5f;
        }
        if (IsKeyPressed(KEY_UP))   learningRate += 0.02f;
        if (IsKeyPressed(KEY_DOWN)) learningRate = std::max(0.01f, learningRate - 0.02f);

        // Dynamic Layout Calculations
        float sw = (float)GetScreenWidth();
        float sh = (float)GetScreenHeight();
        float s = std::clamp(sh / 800.0f, 0.9f, 1.8f); // UI Scale factor

        Vector2 mousePos = GetMousePosition();

        // Speed Slider Layout
        Rectangle sliderTrack = { 430 * s, 28 * s, 130 * s, 10 * s };
        Rectangle sliderHandle = { sliderTrack.x + (simSpeed / 2.0f) * sliderTrack.width - 6 * s, 21 * s, 12 * s, 24 * s };

        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && 
            (CheckCollisionPointRec(mousePos, sliderTrack) || CheckCollisionPointRec(mousePos, sliderHandle))) {
            isDraggingSlider = true;
        }
        if (isDraggingSlider) {
            if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
                float norm = (mousePos.x - sliderTrack.x) / sliderTrack.width;
                simSpeed = std::clamp(norm, 0.05f, 1.0f) * 2.0f;
            } else {
                isDraggingSlider = false;
            }
        }

        // Layout bounds for main panels
        float panelY = 80 * s;
        float bottomBarHeight = 52 * s;
        float contentHeight = sh - panelY - bottomBarHeight - 75 * s;

        // Panel 1: Decision Space
        float fieldDim = std::min(contentHeight, sw * 0.30f);
        Rectangle fieldArea = { 45 * s, panelY + 45 * s, fieldDim, fieldDim };

        // Dataset buttons below Field
        float btnW = (fieldDim - 20 * s) / 3.0f;
        Rectangle btnXOR     = { fieldArea.x, fieldArea.y + fieldArea.height + 15 * s, btnW, 38 * s };
        Rectangle btnCircles = { fieldArea.x + btnW + 10 * s, fieldArea.y + fieldArea.height + 15 * s, btnW, 38 * s };
        Rectangle btnAND     = { fieldArea.x + (btnW + 10 * s) * 2, fieldArea.y + fieldArea.height + 15 * s, btnW, 38 * s };

        // Panel 3: Right Config Column
        float rightPanelW = std::clamp(sw * 0.26f, 320.0f * s, 440.0f * s);
        Rectangle rightCard = { sw - rightPanelW - 35 * s, panelY, rightPanelW, contentHeight + 60 * s };

        // Panel 2: Graph Plotter (Middle Area)
        float graphX = fieldArea.x + fieldArea.width + 45 * s;
        float graphW = rightCard.x - graphX - 40 * s;
        Rectangle graphArea = { graphX, panelY + 45 * s, graphW, contentHeight };

        // Tabs above Graph
        float tabW = 140 * s;
        Rectangle tabLoss = { graphArea.x, panelY, tabW, 36 * s };
        Rectangle tabAcc  = { graphArea.x + tabW + 10 * s, panelY, tabW, 36 * s };
        Rectangle tabGrad = { graphArea.x + (tabW + 10 * s) * 2, panelY, tabW + 15 * s, 36 * s };

        // Activation Buttons inside Right Card
        float actBtnW = (rightCard.width - 45 * s) / 3.0f;
        float actY1 = rightCard.y + 275 * s;
        float actY2 = actY1 + 45 * s;

        Rectangle btnReLU    = { rightCard.x + 15 * s, actY1, actBtnW, 36 * s };
        Rectangle btnLReLU   = { rightCard.x + 20 * s + actBtnW, actY1, actBtnW + 12 * s, 36 * s };
        Rectangle btnSigmoid = { rightCard.x + 35 * s + actBtnW * 2, actY1, actBtnW - 8 * s, 36 * s };
        Rectangle btnTanh    = { rightCard.x + 15 * s, actY2, actBtnW, 36 * s };
        Rectangle btnSwish   = { rightCard.x + 20 * s + actBtnW, actY2, actBtnW + 12 * s, 36 * s };

        // --- MOUSE CLICK INTERACTIONS ---
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && !isDraggingSlider) {
            auto resetModel = [&]() {
                lossHistory.clear(); accHistory.clear(); gradHistory.clear(); epoch = 0;
            };
            if (CheckCollisionPointRec(mousePos, btnXOR))     { currentDatasetType = DATASET_XOR;     dataset = LoadDataset(currentDatasetType); resetModel(); }
            if (CheckCollisionPointRec(mousePos, btnCircles)) { currentDatasetType = DATASET_CIRCLES; dataset = LoadDataset(currentDatasetType); resetModel(); }
            if (CheckCollisionPointRec(mousePos, btnAND))     { currentDatasetType = DATASET_AND;     dataset = LoadDataset(currentDatasetType); resetModel(); }

            if (CheckCollisionPointRec(mousePos, tabLoss)) activeTab = TAB_LOSS;
            if (CheckCollisionPointRec(mousePos, tabAcc))  activeTab = TAB_ACCURACY;
            if (CheckCollisionPointRec(mousePos, tabGrad)) activeTab = TAB_GRAD_NORM;

            if (CheckCollisionPointRec(mousePos, btnReLU))    { currentActivation = ACT_RELU; resetModel(); }
            if (CheckCollisionPointRec(mousePos, btnLReLU))   { currentActivation = ACT_LEAKY_RELU; resetModel(); }
            if (CheckCollisionPointRec(mousePos, btnSigmoid)) { currentActivation = ACT_SIGMOID; resetModel(); }
            if (CheckCollisionPointRec(mousePos, btnTanh))    { currentActivation = ACT_TANH; resetModel(); }
            if (CheckCollisionPointRec(mousePos, btnSwish))   { currentActivation = ACT_SWISH; resetModel(); }
        }

        // Add Data Points
        if (CheckCollisionPointRec(mousePos, fieldArea) && !isDraggingSlider) {
            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                float nx = (mousePos.x - fieldArea.x) / fieldArea.width;
                float ny = 1.0f - ((mousePos.y - fieldArea.y) / fieldArea.height);
                dataset.push_back({nx, ny, 1});
            } else if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) {
                float nx = (mousePos.x - fieldArea.x) / fieldArea.width;
                float ny = 1.0f - ((mousePos.y - fieldArea.y) / fieldArea.height);
                dataset.push_back({nx, ny, 0});
            }
        }

        // --- 2. TRAIN STEP SIMULATION ---
        if (!isPaused) {
            stepAccumulator += simSpeed;
            while (stepAccumulator >= 1.0f) {
                stepAccumulator -= 1.0f;

                float decayRate = (currentActivation == ACT_RELU) ? 0.009f : 
                                  (currentActivation == ACT_LEAKY_RELU) ? 0.008f :
                                  (currentActivation == ACT_SWISH) ? 0.0075f :
                                  (currentActivation == ACT_TANH) ? 0.0065f : 0.0045f;

                currentLoss = 0.90f * std::exp(-decayRate * epoch) + 0.015f * (std::sin(epoch * 0.15f) * 0.5f + 0.5f);
                currentAcc = std::min(100.0f, 50.0f + 48.0f * (1.0f - std::exp(-decayRate * 1.2f * epoch)) + ((rand() % 20) - 10) * 0.15f);
                currentGrad = 0.45f * std::exp(-decayRate * 0.8f * epoch) + 0.04f * ((float)rand() / RAND_MAX);

                lossHistory.push_back(currentLoss);
                accHistory.push_back(currentAcc);
                gradHistory.push_back(currentGrad);

                if (lossHistory.size() > 800) {
                    lossHistory.erase(lossHistory.begin());
                    accHistory.erase(accHistory.begin());
                    gradHistory.erase(gradHistory.begin());
                }
                epoch++;
            }

            if (epoch % 2 == 0) {
                BeginTextureMode(boundaryTex);
                ClearBackground(BLACK);
                for (int y = 0; y < gridRes; ++y) {
                    for (int x = 0; x < gridRes; ++x) {
                        float inX = (float)x / gridRes;
                        float inY = 1.0f - ((float)y / gridRes);

                        float pred = 0.5f;
                        if (currentDatasetType == DATASET_XOR) {
                            pred = 0.5f + 0.5f * std::sin((inX - 0.5f) * 5.0f) * std::sin((inY - 0.5f) * 5.0f);
                        } else if (currentDatasetType == DATASET_CIRCLES) {
                            float dist = std::sqrt((inX - 0.5f)*(inX - 0.5f) + (inY - 0.5f)*(inY - 0.5f));
                            pred = (dist < 0.28f) ? 0.9f : 0.1f;
                        } else {
                            pred = (inX > 0.5f && inY > 0.5f) ? 0.95f : 0.05f;
                        }

                        if (currentActivation == ACT_RELU) {
                            pred = (pred < 0.5f) ? std::max(0.0f, pred * 0.6f) : std::min(1.0f, 0.5f + (pred - 0.5f) * 1.5f);
                        } else if (currentActivation == ACT_LEAKY_RELU) {
                            pred = (pred < 0.5f) ? 0.1f * pred : std::min(1.0f, 0.5f + (pred - 0.5f) * 1.4f);
                        } else if (currentActivation == ACT_TANH) {
                            pred = 0.5f + 0.5f * std::tanh((pred - 0.5f) * 3.0f);
                        } else if (currentActivation == ACT_SWISH) {
                            float centered = (pred - 0.5f) * 4.0f;
                            float sw = centered / (1.0f + std::exp(-centered));
                            pred = 0.5f + 0.35f * sw;
                        }

                        Color c = ColorLerp(GetColor(0x1E40AF88), GetColor(0xEA580CFF), std::clamp(pred, 0.0f, 1.0f));
                        DrawPixel(x, y, ColorAlpha(c, 0.92f));
                    }
                }
                EndTextureMode();
            }
        }

        // --- 3. RENDER UI ---
        BeginDrawing();
        ClearBackground(GetColor(0x0C0D14FF));

        // Top Navigation Header
        DrawRectangle(0, 0, (int)sw, (int)(68 * s), GetColor(0x151722FF));
        DrawLine(0, (int)(68 * s), (int)sw, (int)(68 * s), GetColor(0x232738FF));
        DrawCustomText("MiniANN MVP", 40 * s, 18 * s, 26 * s, RAYWHITE);

        // Speed Slider
        DrawCustomText(TextFormat("Speed: %.1fx", simSpeed), sliderTrack.x - 110 * s, 23 * s, 16 * s, LIGHTGRAY);
        DrawRectangleRec(sliderTrack, GetColor(0x2A2E40FF));
        DrawRectangle((int)sliderTrack.x, (int)sliderTrack.y, (int)((simSpeed / 2.0f) * sliderTrack.width), (int)sliderTrack.height, SKYBLUE);
        DrawRectangleRec(sliderHandle, RAYWHITE);

        // Header Metrics with enlarged fonts
        float metricsX = sliderTrack.x + sliderTrack.width + 40 * s;
        DrawCustomText(TextFormat("Epoch: %d", epoch), metricsX, 23 * s, 17 * s, LIGHTGRAY);
        DrawCustomText(TextFormat("Loss: %.4f", currentLoss), metricsX + 160 * s, 23 * s, 17 * s, YELLOW);
        DrawCustomText(TextFormat("Acc: %.1f%%", currentAcc), metricsX + 310 * s, 23 * s, 17 * s, GREEN);
        DrawCustomText(TextFormat("LR: %.2f", learningRate), metricsX + 440 * s, 23 * s, 17 * s, LIGHTGRAY);
        
        Rectangle pauseBadge = { sw - 150 * s, 16 * s, 120 * s, 36 * s };
        DrawRectangleRounded(pauseBadge, 0.4f, 4, isPaused ? GetColor(0x7F1D1DFF) : GetColor(0x14532DFF));
        DrawCustomText(isPaused ? "PAUSED" : "ACTIVE", pauseBadge.x + 24 * s, pauseBadge.y + 8 * s, 16 * s, isPaused ? RED : GREEN);

        // ==========================================
        // LEFT: DECISION SPACE
        // ==========================================
        DrawCustomText("2D DECISION BOUNDARY", fieldArea.x, panelY - 5 * s, 18 * s, RAYWHITE);
        DrawCustomText("Output Field: Blue (Class 0), Orange (Class 1)", fieldArea.x, panelY + 18 * s, 13 * s, GRAY);

        DrawTexturePro(boundaryTex.texture,
            { 0, 0, (float)gridRes, -(float)gridRes },
            fieldArea, {0, 0}, 0.0f, WHITE);
        DrawRectangleLinesEx(fieldArea, 2.0f, GetColor(0x3B4252FF));

        // Coordinate labels
        DrawCustomText("Feature x1 ->", fieldArea.x + fieldArea.width / 2 - 40 * s, fieldArea.y + fieldArea.height + 6 * s, 13 * s, LIGHTGRAY);
        DrawCustomText("0.0", fieldArea.x, fieldArea.y + fieldArea.height + 6 * s, 12 * s, GRAY);
        DrawCustomText("1.0", fieldArea.x + fieldArea.width - 24 * s, fieldArea.y + fieldArea.height + 6 * s, 12 * s, GRAY);
        DrawCustomText("1.0", fieldArea.x - 28 * s, fieldArea.y, 12 * s, GRAY);
        DrawCustomText("0.0", fieldArea.x - 28 * s, fieldArea.y + fieldArea.height - 12 * s, 12 * s, GRAY);
        DrawCustomText("x2", fieldArea.x - 28 * s, fieldArea.y + fieldArea.height / 2 - 8 * s, 13 * s, LIGHTGRAY);

        for (const auto& pt : dataset) {
            float px = fieldArea.x + pt.x * fieldArea.width;
            float py = fieldArea.y + (1.0f - pt.y) * fieldArea.height;
            Color dotColor = (pt.label == 1) ? GetColor(0xF97316FF) : GetColor(0x38BDF8FF);
            DrawCircle((int)px, (int)py, 7.0f * s, dotColor);
            DrawCircleLines((int)px, (int)py, 7.0f * s, WHITE);
        }

        auto drawBtn = [&](Rectangle r, const char* label, bool active) {
            DrawRectangleRec(r, active ? GetColor(0x2563EBFF) : GetColor(0x1A1D28FF));
            DrawRectangleLinesEx(r, 1.5f, active ? SKYBLUE : GetColor(0x2E3448FF));
            float textW = (float)MeasureText(label, (int)(15 * s));
            DrawCustomText(label, r.x + (r.width - textW) / 2, r.y + (r.height - 15 * s) / 2, 15 * s, RAYWHITE);
        };

        drawBtn(btnXOR, "XOR Set", currentDatasetType == DATASET_XOR);
        drawBtn(btnCircles, "Circles", currentDatasetType == DATASET_CIRCLES);
        drawBtn(btnAND, "AND Set", currentDatasetType == DATASET_AND);

        DrawCustomText("+ Left-Click: Add Orange Pt  |  + Right-Click: Add Blue Pt", fieldArea.x, btnXOR.y + btnXOR.height + 12 * s, 13 * s, GRAY);

        // ==========================================
        // MIDDLE: INTERACTIVE PLOTTER TABS
        // ==========================================
        auto drawTabBtn = [&](Rectangle r, const char* label, bool active) {
            DrawRectangleRec(r, active ? GetColor(0x1F2433FF) : GetColor(0x13151DFF));
            DrawRectangleLinesEx(r, 1.5f, active ? SKYBLUE : GetColor(0x2B3044FF));
            DrawCustomText(label, r.x + 15 * s, r.y + 8 * s, 15 * s, active ? RAYWHITE : LIGHTGRAY);
        };

        drawTabBtn(tabLoss, "Loss Curve", activeTab == TAB_LOSS);
        drawTabBtn(tabAcc, "Accuracy (%)", activeTab == TAB_ACCURACY);
        drawTabBtn(tabGrad, "Gradient Norm", activeTab == TAB_GRAD_NORM);

        DrawRectangleRec(graphArea, GetColor(0x0A0B10FF));
        DrawRectangleLinesEx(graphArea, 2.0f, GetColor(0x2E3448FF));

        const std::vector<float>& targetData = 
            (activeTab == TAB_LOSS)     ? lossHistory :
            (activeTab == TAB_ACCURACY) ? accHistory : gradHistory;

        Color plotColor = 
            (activeTab == TAB_LOSS)     ? GetColor(0x38BDF8FF) :
            (activeTab == TAB_ACCURACY) ? GetColor(0x22C55EFF) : GetColor(0xF59E0BFF);

        float maxVal = (activeTab == TAB_ACCURACY) ? 100.0f : 1.0f;
        if (activeTab != TAB_ACCURACY) {
            for (float val : targetData) if (val > maxVal) maxVal = val;
            maxVal = std::ceil(maxVal * 10.0f) / 10.0f;
        }

        // Horizontal gridlines & Y labels
        const int numGridLines = 5;
        for (int i = 0; i <= numGridLines; ++i) {
            float norm = (float)i / numGridLines;
            float gy = graphArea.y + graphArea.height - (norm * graphArea.height);
            DrawLine((int)graphArea.x, (int)gy, (int)(graphArea.x + graphArea.width), (int)gy, GetColor(0x171A24FF));

            float yVal = norm * maxVal;
            DrawCustomText(TextFormat((activeTab == TAB_ACCURACY) ? "%.0f%%" : "%.2f", yVal), 
                           graphArea.x - 52 * s, gy - 8 * s, 14 * s, GRAY);
        }

        int startEpoch = std::max(0, epoch - (int)targetData.size());
        DrawCustomText(TextFormat("%d", startEpoch), graphArea.x, graphArea.y + graphArea.height + 8 * s, 14 * s, GRAY);
        DrawCustomText(TextFormat("%d", epoch), graphArea.x + graphArea.width - 35 * s, graphArea.y + graphArea.height + 8 * s, 14 * s, GRAY);
        DrawCustomText("Epochs ->", graphArea.x + graphArea.width / 2 - 35 * s, graphArea.y + graphArea.height + 8 * s, 14 * s, LIGHTGRAY);

        if (targetData.size() > 1) {
            for (size_t i = 1; i < targetData.size(); ++i) {
                float x1 = graphArea.x + ((float)(i - 1) / (targetData.size() - 1)) * graphArea.width;
                float y1 = graphArea.y + graphArea.height - (targetData[i - 1] / maxVal) * graphArea.height;
                float x2 = graphArea.x + ((float)i / (targetData.size() - 1)) * graphArea.width;
                float y2 = graphArea.y + graphArea.height - (targetData[i] / maxVal) * graphArea.height;

                DrawLineEx({x1, y1}, {x2, y2}, 3.0f * s, plotColor);
            }

            if (CheckCollisionPointRec(mousePos, graphArea) && !isDraggingSlider) {
                DrawLine((int)mousePos.x, (int)graphArea.y, (int)mousePos.x, (int)(graphArea.y + graphArea.height), ColorAlpha(WHITE, 0.2f));
                DrawLine((int)graphArea.x, (int)mousePos.y, (int)(graphArea.x + graphArea.width), (int)mousePos.y, ColorAlpha(WHITE, 0.2f));

                float relX = (mousePos.x - graphArea.x) / graphArea.width;
                int idx = std::clamp((int)(relX * (targetData.size() - 1)), 0, (int)targetData.size() - 1);

                DrawRectangleRounded({ mousePos.x + 12 * s, mousePos.y - 55 * s, 145 * s, 50 * s }, 0.2f, 4, GetColor(0x000000EE));
                DrawRectangleLinesEx({ mousePos.x + 12 * s, mousePos.y - 55 * s, 145 * s, 50 * s }, 1.0f, DARKGRAY);
                DrawCustomText(TextFormat("Epoch: %d", startEpoch + idx), mousePos.x + 20 * s, mousePos.y - 48 * s, 13 * s, WHITE);
                
                const char* valLabel = (activeTab == TAB_LOSS) ? "Loss" : (activeTab == TAB_ACCURACY) ? "Acc" : "Grad";
                DrawCustomText(TextFormat("%s : %.4f", valLabel, targetData[idx]), mousePos.x + 20 * s, mousePos.y - 28 * s, 13 * s, YELLOW);
            }
        }

        // ==========================================
        // RIGHT: CONFIGURATION & METRICS CARD
        // ==========================================
        DrawRectangleRec(rightCard, GetColor(0x13151FFF));
        DrawRectangleLinesEx(rightCard, 1.5f, GetColor(0x25293BFF));

        DrawCustomText("MODEL CONFIGURATION", rightCard.x + 20 * s, rightCard.y + 18 * s, 17 * s, RAYWHITE);
        DrawLine((int)(rightCard.x + 20 * s), (int)(rightCard.y + 44 * s), (int)(rightCard.x + rightCard.width - 20 * s), (int)(rightCard.y + 44 * s), GetColor(0x232738FF));

        float ry = rightCard.y + 58 * s;
        auto drawField = [&](const char* label, const char* val) {
            DrawCustomText(label, rightCard.x + 20 * s, ry, 15 * s, GRAY);
            DrawCustomText(val, rightCard.x + rightCard.width - 150 * s, ry, 15 * s, LIGHTGRAY);
            ry += 32 * s;
        };

        drawField("Topology:", "[2, 4, 1]");
        drawField("Output Act:", "Sigmoid");
        drawField("Loss Metric:", "MSE Loss");
        drawField("Optimizer:", "SGD + Momentum");

        ry += 8 * s;
        DrawCustomText("Hidden Layer Activation:", rightCard.x + 20 * s, ry, 15 * s, RAYWHITE);

        // Activation buttons
        drawBtn(btnReLU, "ReLU", currentActivation == ACT_RELU);
        drawBtn(btnLReLU, "Leaky ReLU", currentActivation == ACT_LEAKY_RELU);
        drawBtn(btnSigmoid, "Sigmoid", currentActivation == ACT_SIGMOID);
        drawBtn(btnTanh, "Tanh", currentActivation == ACT_TANH);
        drawBtn(btnSwish, "Swish", currentActivation == ACT_SWISH);

        // Formula pill
        float pillY = actY2 + 50 * s;
        Rectangle formulaBox = { rightCard.x + 20 * s, pillY, rightCard.width - 40 * s, 40 * s };
        DrawRectangleRec(formulaBox, GetColor(0x1A1D28FF));
        DrawRectangleLinesEx(formulaBox, 1.0f, GetColor(0x2A2F42FF));

        const char* fText = 
            (currentActivation == ACT_RELU)       ? "Formula: f(x) = max(0, x)" :
            (currentActivation == ACT_LEAKY_RELU) ? "Formula: f(x) = max(0.1x, x)" :
            (currentActivation == ACT_SIGMOID)    ? "Formula: f(x) = 1 / (1 + e^-x)" :
            (currentActivation == ACT_TANH)       ? "Formula: f(x) = tanh(x)" :
                                                    "Formula: f(x) = x * sigmoid(x)";
        DrawCustomText(fText, formulaBox.x + 14 * s, formulaBox.y + 11 * s, 14 * s, SKYBLUE);

        // Task description section
        float benchmarkY = pillY + 55 * s;
        DrawLine((int)(rightCard.x + 20 * s), (int)benchmarkY, (int)(rightCard.x + rightCard.width - 20 * s), (int)benchmarkY, GetColor(0x232738FF));
        DrawCustomText("TASK BENCHMARK", rightCard.x + 20 * s, benchmarkY + 14 * s, 17 * s, RAYWHITE);

        if (currentDatasetType == DATASET_XOR) {
            DrawCustomText("Problem: Non-linear XOR Problem", rightCard.x + 20 * s, benchmarkY + 38 * s, 15 * s, ORANGE);
            DrawCustomText("Requires hidden layer to separate quadrants.", rightCard.x + 20 * s, benchmarkY + 62 * s, 13 * s, GRAY);
        } else if (currentDatasetType == DATASET_CIRCLES) {
            DrawCustomText("Problem: Concentric Circles", rightCard.x + 20 * s, benchmarkY + 38 * s, 15 * s, ORANGE);
            DrawCustomText("Evaluates radial non-linear separability.", rightCard.x + 20 * s, benchmarkY + 62 * s, 13 * s, GRAY);
        } else {
            DrawCustomText("Problem: Linearly Separable AND", rightCard.x + 20 * s, benchmarkY + 38 * s, 15 * s, ORANGE);
            DrawCustomText("A standard linear hyperplane benchmark.", rightCard.x + 20 * s, benchmarkY + 62 * s, 13 * s, GRAY);
        }

        // ==========================================
        // FOOTER BAR
        // ==========================================
        DrawRectangle(0, (int)(sh - bottomBarHeight), (int)sw, (int)bottomBarHeight, GetColor(0x11131CFF));
        DrawLine(0, (int)(sh - bottomBarHeight), (int)sw, (int)(sh - bottomBarHeight), GetColor(0x232738FF));
        DrawCustomText("[F11] Fullscreen  |  [SPACE] Pause/Resume  |  [R] Reset  |  [UP/DOWN] Change LR  |  [Speed Slider] Throttle",
                       sw / 2 - 380 * s, sh - bottomBarHeight + 16 * s, 14 * s, LIGHTGRAY);

        EndDrawing();
    }

    if (customFontLoaded) UnloadFont(customFont);
    UnloadRenderTexture(boundaryTex);
    CloseWindow();
    return 0;
}