// MiniANN MVP - Qt Widgets Network Workbench (spec-driven UI layer).
//
// Architecture (spec §35/§36):
//   UI -> ExperimentConfig -> ExperimentController -> ANN backend
//                                                   -> ExperimentResult -> UI
// This file contains NO training math and NO dataset parsing: it builds an
// ExperimentConfig from the widgets, runs it on a worker thread through
// ExperimentController::run(), and visualizes the streamed + final metrics.
// No Q_OBJECT usage anywhere, so it builds with plain g++ (no moc).

#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QComboBox>
#include <QCheckBox>
#include <QFrame>
#include <QTimer>
#include <QShortcut>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QDialog>
#include <QDialogButtonBox>
#include <QTableWidget>
#include <QHeaderView>
#include <QFileDialog>
#include <QMessageBox>
#include <QPainter>
#include <QPainterPath>
#include <QPaintEvent>
#include <QMouseEvent>
#include <QToolTip>
#include <QFontMetrics>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QCloseEvent>
#include <QMimeData>
#include <QUrl>
#include <QGuiApplication>
#include <QScreen>
#include <QStyleFactory>
#include <QScrollArea>
#include <QFile>
#include <QTextStream>
#include <QFileInfo>

#include "miniann/experiment.hpp"
#include "miniann/serializer.hpp"

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
#include <cstdio>

using namespace miniann;

enum ActiveDataset { DS_AND = 0, DS_OR = 1, DS_XOR = 2, DS_IRIS = 3, DS_CSV = 4 };
enum ActiveActivation { ACT_SIGMOID = 0, ACT_TANH = 1, ACT_RELU = 2, ACT_LEAKY_RELU = 3, ACT_SWISH = 4 };
enum ActiveLoss { LOSS_MSE = 0, LOSS_BCE = 1, LOSS_CCE = 2 };
enum ActiveOptimizer { OPT_SGD = 0, OPT_MOMENTUM = 1, OPT_ADAM = 2 };
enum ViewMode { VIEW_LOSS = 0, VIEW_ACC = 1, VIEW_BOUNDARY = 2, VIEW_NETWORK = 3 };

static const char* kDsNames[5] = {"AND", "OR", "XOR", "Iris", "CSV"};
static const char* kDsKeys[5] = {"and", "or", "xor", "iris", "csv"};
static const char* kActLabels[5] = {"Sig", "Tanh", "ReLU", "L-ReLU", "Swish"};
static const char* kActKeys[5] = {"sigmoid", "tanh", "relu", "leaky_relu", "swish"};
static const char* kLossKeys[3] = {"mse", "bce", "cce"};
static const char* kLossLabels[3] = {"MSE", "BCE", "CCE"};
static const char* kOptKeys[3] = {"sgd", "momentum", "adam"};
static const char* kOptLabels[3] = {"SGD", "Mom.", "Adam"};
static const char* kOutActKeys[6] = {"sigmoid", "tanh", "relu", "leaky_relu", "swish", "linear"};
static const char* kOutActLabels[6] = {"Sigmoid", "Tanh", "ReLU", "L-ReLU", "Swish", "Linear"};
static const std::size_t kBatchVals[7] = {1, 8, 16, 32, 64, 128, 0};
static const char* kBatchLabels[7] = {"1", "8", "16", "32", "64", "128", "Full"};
static const char* kClassColors[6] = {"#3B82F6", "#EF4444", "#22C55E", "#F59E0B", "#A855F7", "#14B8A6"};

static std::string actToName(ActiveActivation a) { return kActKeys[(int)a]; }
static ActiveActivation actFromName(const std::string& n) {
    for (int a = 0; a < 5; ++a)
        if (n == kActKeys[a]) return ActiveActivation(a);
    throw std::invalid_argument("unsupported hidden activation '" + n + "'");
}

// ---------------------------------------------------------------- live bridge
// Streaming sink for the worker thread + store for completed-run artifacts.
// The GUI thread only ever reads under mtx (or atomics); the worker only
// writes under mtx. Culturear curves stream per epoch; the heavy objects
// (network, datasets) are stored once at the end of a run.
struct LiveGuiBridge : public TrainingCallback {
    std::mutex mtx;
    std::vector<float> trainLoss, valLoss, trainAcc, valAcc;
    int currentEpoch = 0;
    std::atomic<bool> isTraining{false};
    std::atomic<bool> stopRequested{false};
    std::string summary = "Ready to train.";
    // -- completed-run snapshot --
    unsigned runSeq = 0;
    bool hasResult = false;
    bool lastCompleted = false;
    bool lastStopped = false;
    std::string lastError;
    double lastTrainLoss = 0, lastTrainAcc = 0, lastValLoss = 0, lastValAcc = 0;
    double lastTestLoss = 0, lastTestAcc = 0, lastSeconds = 0;
    int lastEpochsRun = 0, lastEpochsTarget = 0;
    bool lastHasVal = false, lastHasTest = false, lastTiny = false;
    std::string lastWarning, lastArch, lastOptDesc, lastDsDesc, lastLoss;
    std::string lastBatchTxt;
    int lastSeed = 42;
    std::vector<std::vector<std::size_t>> lastConfusion;
    std::size_t lastNumClasses = 0;
    bool lastHasConfusion = false, lastConfusionOnTrain = false;
    std::string lastSplitTxt;
    std::shared_ptr<NeuralNetwork> lastNet;
    Dataset lastTrain, lastTest;
    std::size_t lastInDim = 0, lastOutDim = 0;

    void onEpoch(int epoch, const TrainingHistory& hist) override {
        std::lock_guard<std::mutex> lock(mtx);
        currentEpoch = epoch;
        if (!hist.trainLoss.empty())       trainLoss.push_back((float)hist.trainLoss.back());
        if (!hist.validationLoss.empty())  valLoss.push_back((float)hist.validationLoss.back());
        if (!hist.trainAcc.empty())        trainAcc.push_back((float)hist.trainAcc.back());
        if (!hist.validationAcc.empty())   valAcc.push_back((float)hist.validationAcc.back());
    }
    bool shouldStop() const override { return stopRequested.load(); }

    void resetLive() {
        std::lock_guard<std::mutex> lock(mtx);
        trainLoss.clear();
        valLoss.clear();
        trainAcc.clear();
        valAcc.clear();
        currentEpoch = 0;
        summary = "Ready to train.";
        hasResult = false;
        lastError.clear();
        lastCompleted = false;
        lastStopped = false;
    }
};

static LiveGuiBridge g_bridge;
static std::unique_ptr<std::thread> g_trainThread = nullptr;

// ---------------------------------------------------------------- plot widget
class PlotWidget : public QWidget {
public:
    explicit PlotWidget(QWidget* parent = nullptr) : QWidget(parent) {
        setMouseTracking(true);
        setMinimumHeight(200);
    }

    void setCurves(std::vector<float> train, std::vector<float> val, bool accuracyMode,
                   bool hasVal, int liveEpoch) {
        m_train = std::move(train);
        m_val = std::move(val);
        m_accuracy = accuracyMode;
        m_hasVal = hasVal && !m_val.empty();
        m_liveEpoch = liveEpoch;
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing, true);
        p.fillRect(rect(), QColor(0x09, 0x0A, 0x0F));

        const int ml = 52, mr = 14, mt = 32, mb = 28;
        QRect area(ml, mt, width() - ml - mr, height() - mt - mb);
        if (area.width() < 40 || area.height() < 40) return;

        p.setPen(QPen(QColor(0x2E, 0x34, 0x48), 1));
        p.setBrush(Qt::NoBrush);
        p.drawRect(area.adjusted(0, 0, -1, -1));

        const std::vector<float>& main = m_train;
        float maxVal = 1.0f;
        for (float v : m_train) if (v > maxVal) maxVal = v;
        if (m_hasVal)
            for (float v : m_val) if (v > maxVal) maxVal = v;
        if (!m_accuracy) maxVal = std::ceil(maxVal * 10.0f) / 10.0f;
        if (maxVal <= 0.0f) maxVal = 1.0f;

        QFont tickFont = font();
        tickFont.setPointSize(8);
        p.setFont(tickFont);
        p.setPen(QPen(QColor(0x18, 0x1B, 0x26), 1));
        for (int i = 0; i <= 5; ++i) {
            float norm = (float)i / 5.0f;
            int gy = area.bottom() - int(norm * area.height());
            p.drawLine(area.left(), gy, area.right(), gy);
            float yVal = norm * maxVal;
            QString txt = m_accuracy ? QString::asprintf("%.0f%%", yVal * 100.0f)
                                     : QString::asprintf("%.2f", yVal);
            p.setPen(QColor(0xA8, 0xAE, 0xBD));
            p.drawText(QRect(0, gy - 10, ml - 6, 20), Qt::AlignRight | Qt::AlignVCenter, txt);
            p.setPen(QPen(QColor(0x18, 0x1B, 0x26), 1));
        }

        p.setPen(QColor(0xA8, 0xAE, 0xBD));
        p.drawText(QRect(area.left(), area.bottom() + 4, 60, 20), Qt::AlignLeft | Qt::AlignTop, QStringLiteral("0"));
        QString maxTxt = QString::number(m_liveEpoch);
        int maxW = fontMetrics().horizontalAdvance(maxTxt);
        p.drawText(QRect(area.right() - maxW, area.bottom() + 4, maxW, 20), Qt::AlignRight | Qt::AlignTop, maxTxt);
        p.setPen(QColor(0xD0, 0xD3, 0xDB));
        p.drawText(QRect(area.left(), area.bottom() + 4, area.width(), 20), Qt::AlignHCenter | Qt::AlignTop, QStringLiteral("Epochs ->"));

        auto toPt = [&](std::size_t i, std::size_t n, float v) {
            double x = (n < 2) ? area.left() : area.left() + (double)i / (double)(n - 1) * area.width();
            double y = area.bottom() - std::min(1.0, std::max(0.0, (double)v / maxVal)) * area.height();
            return QPointF(x, y);
        };
        auto stride = [](std::size_t n) -> std::size_t {
            if (n <= 2000) return 1;
            return (n + 1999) / 2000;
        };

        if (m_hasVal && m_val.size() > 1) {
            QPainterPath path;
            std::size_t st = stride(m_val.size());
            path.moveTo(toPt(0, m_val.size(), m_val[0]));
            for (std::size_t i = st; i < m_val.size(); i += st) path.lineTo(toPt(i, m_val.size(), m_val[i]));
            p.setPen(QPen(QColor(0xFF, 0xA5, 0x00), 2));
            p.drawPath(path);
        }
        if (main.size() > 1) {
            QPainterPath path;
            std::size_t st = stride(main.size());
            path.moveTo(toPt(0, main.size(), main[0]));
            for (std::size_t i = st; i < main.size(); i += st) path.lineTo(toPt(i, main.size(), main[i]));
            QColor c = m_accuracy ? QColor(0x22, 0xC5, 0x5E) : QColor(0x87, 0xCE, 0xEB);
            p.setPen(QPen(c, 2.2));
            p.drawPath(path);
        }

        // Legend: validation entry only exists when there is validation data (§31).
        QFont legFont = font();
        legFont.setPointSize(9);
        p.setFont(legFont);
        QFontMetrics lfm(legFont);
        QString t1 = QStringLiteral("Train"), t2 = QStringLiteral("Validation");
        if (!m_train.empty()) t1 += QStringLiteral(" ") + fmtVal(m_train.back());
        if (m_hasVal && !m_val.empty()) t2 += QStringLiteral(" ") + fmtVal(m_val.back());
        int w1 = lfm.horizontalAdvance(t1), w2 = lfm.horizontalAdvance(t2);
        int dotR = 5, gap = 6, gapEntries = 22;
        int totalW = dotR * 2 + gap + w1 + (m_hasVal ? gapEntries + dotR * 2 + gap + w2 : 0);
        int lx = area.right() - 10 - totalW;
        int dotY = area.top() + 12;
        QColor c1 = m_accuracy ? QColor(0x22, 0xC5, 0x5E) : QColor(0x87, 0xCE, 0xEB);
        p.setBrush(c1);
        p.setPen(Qt::NoPen);
        p.drawEllipse(QPoint(lx + dotR, dotY), dotR, dotR);
        p.setPen(QColor(0xD0, 0xD3, 0xDB));
        p.drawText(lx + dotR * 2 + gap, dotY + lfm.ascent() / 2 - 5, w1 + 20, 20, Qt::AlignLeft | Qt::AlignTop, t1);
        if (m_hasVal) {
            int lx2 = lx + dotR * 2 + gap + w1 + gapEntries;
            p.setBrush(QColor(0xFF, 0xA5, 0x00));
            p.setPen(Qt::NoPen);
            p.drawEllipse(QPoint(lx2 + dotR, dotY), dotR, dotR);
            p.setPen(QColor(0xD0, 0xD3, 0xDB));
            p.drawText(lx2 + dotR * 2 + gap, dotY + lfm.ascent() / 2 - 5, w2 + 20, 20, Qt::AlignLeft | Qt::AlignTop, t2);
        }

        if (m_hover >= 0 && main.size() > 1) {
            double hx = area.left() + (double)m_hover / (double)(main.size() - 1) * area.width();
            p.setPen(QPen(QColor(255, 255, 255, 64), 1));
            p.drawLine(QPointF(hx, area.top()), QPointF(hx, area.bottom()));
        }
    }

    void mouseMoveEvent(QMouseEvent* e) override {
        const int ml = 52, mr = 14, mt = 32, mb = 28;
        QRect area(ml, mt, width() - ml - mr, height() - mt - mb);
        m_hover = -1;
        if (!m_train.empty() && area.contains(e->pos())) {
            double rel = (double)(e->pos().x() - area.left()) / (double)std::max(1, area.width());
            int idx = (int)(rel * (m_train.size() - 1));
            idx = std::max(0, std::min((int)m_train.size() - 1, idx));
            m_hover = idx;
            const char* k1 = m_accuracy ? "Train Acc" : "Train Loss";
            const char* k2 = m_accuracy ? "Val Acc" : "Val Loss";
            QString tip = QString::asprintf("Epoch: %d", idx + 1)
                + QStringLiteral("\n") + QString::fromLatin1(k1) + QStringLiteral(": ") + fmtVal(m_train[(std::size_t)idx]);
            if (m_hasVal && idx < (int)m_val.size())
                tip += QStringLiteral("\n") + QString::fromLatin1(k2) + QStringLiteral(": ") + fmtVal(m_val[(std::size_t)idx]);
            QToolTip::showText(e->globalPosition().toPoint(), tip, this);
            update();
        } else {
            QToolTip::hideText();
            update();
        }
    }

    void leaveEvent(QEvent*) override {
        m_hover = -1;
        QToolTip::hideText();
        update();
    }

private:
    QString fmtVal(float v) const {
        return m_accuracy ? QString::asprintf("%.1f%%", v * 100.0f) : QString::asprintf("%.4f", v);
    }
    std::vector<float> m_train, m_val;
    bool m_accuracy = false, m_hasVal = false;
    int m_liveEpoch = 0;
    int m_hover = -1;
};

// ------------------------------------------------------- decision boundary
class BoundaryWidget : public QWidget {
public:
    explicit BoundaryWidget(QWidget* parent = nullptr) : QWidget(parent) {
        setMinimumHeight(200);
    }

    void setData(QImage grid, std::vector<QPointF> trainPts, std::vector<int> trainCls,
                 std::vector<QPointF> testPts, std::vector<int> testCls,
                 double x0, double x1, double y0, double y1, bool ready) {
        m_grid = std::move(grid);
        m_trainPts = std::move(trainPts);
        m_trainCls = std::move(trainCls);
        m_testPts = std::move(testPts);
        m_testCls = std::move(testCls);
        m_x0 = x0; m_x1 = x1; m_y0 = y0; m_y1 = y1;
        m_ready = ready;
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing, true);
        p.fillRect(rect(), QColor(0x09, 0x0A, 0x0F));
        const int m = 14;
        QRect area(m, m + 16, width() - 2 * m, height() - 2 * m - 22);
        if (area.width() < 40 || area.height() < 40) return;
        p.setPen(QPen(QColor(0x2E, 0x34, 0x48), 1));
        p.setBrush(Qt::NoBrush);
        p.drawRect(area.adjusted(0, 0, -1, -1));
        if (!m_ready || m_grid.isNull()) {
            p.setPen(QColor(0x8A, 0x90, 0xA0));
            p.drawText(area, Qt::AlignCenter,
                       QStringLiteral("Train a 2-feature dataset (AND / OR / XOR)\nto see its decision boundary."));
            return;
        }
        p.drawImage(area, m_grid);
        auto dot = [&](QPointF d, int c, bool test) {
            double px = area.left() + (d.x() - m_x0) / (m_x1 - m_x0) * area.width();
            double py = area.bottom() - (d.y() - m_y0) / (m_y1 - m_y0) * area.height();
            QColor col(kClassColors[c % 6]);
            p.setBrush(col);
            p.setPen(QPen(Qt::white, test ? 2 : 1));
            if (test) {
                QRectF r(px - 6, py - 6, 12, 12);
                p.drawRect(r);
                p.setBrush(Qt::NoBrush);
                p.setPen(QPen(col, 2));
                p.drawRect(r);
            } else {
                p.drawEllipse(QPointF(px, py), 6, 6);
            }
        };
        for (std::size_t i = 0; i < m_trainPts.size(); ++i) dot(m_trainPts[i], m_trainCls[i], false);
        for (std::size_t i = 0; i < m_testPts.size(); ++i) dot(m_testPts[i], m_testCls[i], true);

        p.setPen(QColor(0xA8, 0xAE, 0xBD));
        QFont f = font();
        f.setPointSize(8);
        p.setFont(f);
        p.drawText(QRect(area.left(), area.bottom() + 4, area.width(), 18), Qt::AlignHCenter,
                   QStringLiteral("x1 ->                                   x2 ^   (rings = test points)"));
    }

private:
    QImage m_grid;
    std::vector<QPointF> m_trainPts, m_testPts;
    std::vector<int> m_trainCls, m_testCls;
    double m_x0 = 0, m_x1 = 1, m_y0 = 0, m_y1 = 1;
    bool m_ready = false;
};

// ------------------------------------------------------- network diagram
struct ArchDesc {
    bool valid = false;
    std::vector<std::size_t> sizes;
    std::vector<QString> names; // per layer: "Input", "Hidden 1 (ReLU)", ...
};

class NetWidget : public QWidget {
public:
    explicit NetWidget(QWidget* parent = nullptr) : QWidget(parent) {
        setMinimumHeight(200);
    }
    void setArch(ArchDesc a) {
        m_arch = std::move(a);
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing, true);
        p.fillRect(rect(), QColor(0x09, 0x0A, 0x0F));
        const int m = 14;
        QRect area(m, m + 8, width() - 2 * m, height() - 2 * m - 40);
        if (area.width() < 40 || area.height() < 40) return;
        p.setPen(QPen(QColor(0x2E, 0x34, 0x48), 1));
        p.setBrush(Qt::NoBrush);
        p.drawRect(area.adjusted(0, 0, -1, -1));
        if (!m_arch.valid || m_arch.sizes.size() < 2) {
            p.setPen(QColor(0x8A, 0x90, 0xA0));
            p.drawText(area, Qt::AlignCenter,
                       QStringLiteral("Train a network (or load a model)\nto see its architecture diagram."));
            return;
        }
        std::size_t L = m_arch.sizes.size();
        std::vector<std::vector<QPointF>> pts(L);
        std::vector<std::size_t> shown(L);
        for (std::size_t l = 0; l < L; ++l) {
            shown[l] = std::min<std::size_t>(m_arch.sizes[l], 12);
            double cx = area.left() + (L == 1 ? area.width() / 2.0
                                              : (double)l / (double)(L - 1) * area.width());
            for (std::size_t i = 0; i < shown[l]; ++i) {
                double cy = (shown[l] == 1) ? area.center().y()
                    : area.top() + 18 + (double)i / (double)(shown[l] - 1) * (area.height() - 36);
                pts[l].push_back(QPointF(cx, cy));
            }
        }
        // Edges (capped: draw every k-th when huge).
        std::size_t edges = 0;
        for (std::size_t l = 1; l < L; ++l) edges += shown[l - 1] * shown[l];
        std::size_t k = edges > 800 ? (edges + 799) / 800 : 1;
        p.setPen(QPen(QColor(0x5A, 0xA9, 0xE6, 70), 1));
        std::size_t e = 0;
        for (std::size_t l = 1; l < L; ++l)
            for (auto a : pts[l - 1])
                for (auto b : pts[l]) {
                    if ((e++ % k) == 0) p.drawLine(a, b);
                }
        for (std::size_t l = 0; l < L; ++l) {
            for (auto c : pts[l]) {
                p.setBrush(QColor(0x1B, 0x1E, 0x2B));
                p.setPen(QPen(QColor(0x7F, 0xB3, 0xE8), 1.5));
                p.drawEllipse(c, 9, 9);
            }
            if (shown[l] < m_arch.sizes[l]) {
                p.setPen(QColor(0x8A, 0x90, 0xA0));
                QFont f = font();
                f.setPointSize(8);
                p.setFont(f);
                double cx = pts[l][0].x();
                p.drawText(QRect(int(cx) - 40, area.bottom() - 34, 80, 16), Qt::AlignHCenter,
                           QStringLiteral("+") + QString::number((unsigned long long)(m_arch.sizes[l] - shown[l])));
            }
        }
        QFont lf = font();
        lf.setPointSize(9);
        lf.setBold(true);
        p.setFont(lf);
        p.setPen(QColor(0xD0, 0xD3, 0xDB));
        for (std::size_t l = 0; l < L; ++l) {
            double cx = (L == 1) ? area.center().x()
                                 : area.left() + (double)l / (double)(L - 1) * area.width();
            QString t = m_arch.names[l] + QString::asprintf(" [%llu]", (unsigned long long)m_arch.sizes[l]);
            int w = fontMetrics().horizontalAdvance(t);
            p.drawText(QRect(int(cx) - w / 2, area.bottom() + 4, w + 4, 18), Qt::AlignLeft, t);
        }
    }

private:
    ArchDesc m_arch;
};

// ---------------------------------------------------------------- main window
class MainWindow : public QMainWindow {
public:
    enum UiState { ST_IDLE, ST_READY, ST_TRAINING, ST_COMPLETED, ST_STOPPED, ST_ERROR };

    explicit MainWindow(QWidget* parent = nullptr) : QMainWindow(parent) {
        setWindowTitle(QStringLiteral("MiniANN MVP - Interactive Network Workbench"));
        setMinimumSize(1060, 740);
        resize(1420, 910);
        setAcceptDrops(true);

        auto* shortcut = new QShortcut(QKeySequence(Qt::Key_F11), this);
        connect(shortcut, &QShortcut::activated, [this]() {
            isMaximized() ? showNormal() : showMaximized();
        });

        // Stylesheet FIRST (before any child exists), so every widget is
        // polished with the final style from birth — same order as the
        // isolated probe, which renders perfectly. --noqss skips it (debug).
        if (!QApplication::arguments().contains(QStringLiteral("--noqss")))
            applyTheme();
        buildUi();
        applyPresetXor(false);
        // Debug hook: preselect CSV (empty path) to screenshot the upload row.
        if (QApplication::arguments().contains(QStringLiteral("--csv")))
            m_dataset = DS_CSV;
        refreshConfigUi();

        m_timer = new QTimer(this);
        connect(m_timer, &QTimer::timeout, [this]() { refreshLiveUi(); });
        // --notimer freezes the 10 Hz refresh (debug isolation for paint issues).
        if (!QApplication::arguments().contains(QStringLiteral("--notimer")))
            m_timer->start(100);
        refreshLiveUi();
    }

    // Test hook for --dump: snapshots widget state researchers can't see.
    void dumpDebug(const QString& path) {
        QFile f(path);
        if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) return;
        QTextStream out(&f);
        out << "appFont=" << QApplication::font().family()
            << " pt=" << QApplication::font().pointSize() << "\n";
        auto dumpBtn = [&](const char* tag, QPushButton* b) {
            out << tag << " text=[" << b->text() << "] enabled=" << b->isEnabled()
                << " enabledToMain=" << b->isEnabledTo(this) << " visible=" << b->isVisible()
                << " font=" << b->font().family() << " pt=" << b->font().pointSize()
                << " size=" << b->size().width() << "x" << b->size().height() << "\n";
        };
        if (!m_dsBtns.empty()) dumpBtn("ds0", m_dsBtns[0]);
        if (m_dsBtns.size() > 2) dumpBtn("ds2", m_dsBtns[2]);
        dumpBtn("train", m_trainBtn);
        dumpBtn("tabLoss", m_tabLoss);
        out << "configBox enabled=" << m_configBox->isEnabled() << "\n";
        out << "state=" << m_state << " previewErr=[" << QString::fromStdString(m_previewErr) << "]\n";
        out << "devicePixelRatio=" << devicePixelRatio() << "\n";
        out << "dsInfo=[" << m_dsInfoLbl->text() << "]\n";
        out << "cfg=[" << m_cfgLbl->text() << "]\n";
        out << "arch=[" << m_archLbl->text() << "]\n";
        out << "layers=[" << m_layersLbl->text() << "]\n";
        auto geo = [&](const char* tag, QWidget* w) {
            QRect g = w->geometry();
            out << tag << " geo=" << g.x() << "," << g.y() << "," << g.width() << "x" << g.height()
                << " vis=" << w->isVisible() << " min=" << w->minimumSize().width() << "x"
                << w->minimumSize().height() << " hint=" << w->sizeHint().width() << "x"
                << w->sizeHint().height() << "\n";
        };
        geo("dsInfo", m_dsInfoLbl);
        geo("cfg", m_cfgLbl);
        geo("summary", m_summaryLbl);
        geo("arch", m_archLbl);
        geo("warn", m_warnLbl);
        if (!m_layerRows[0].acts.empty()) geo("act0", m_layerRows[0].acts[0]);
        {
            int wsum = 0;
            for (auto* b : m_layerRows[0].acts) wsum += b->width();
            out << "h1acts totalW=" << wsum << " rowW=" << m_layerRows[0].row->width() << "\n";
        }
        QString dir = QFileInfo(path).absolutePath() + QStringLiteral("/");
        if (!m_dsBtns.empty()) m_dsBtns[0]->grab().save(dir + QStringLiteral("w_ds.png"));
        if (!m_dsBtns.empty()) m_dsBtns.back()->grab().save(dir + QStringLiteral("w_csv.png"));// last button (edge!)
        if (!m_layerRows[0].acts.empty()) {
            m_layerRows[0].acts[1]->grab().save(dir + QStringLiteral("w_act.png"));
            m_layerRows[0].acts.back()->grab().save(dir + QStringLiteral("w_swish.png")); // last button (edge!)
            auto* spin = m_layerRows[0].size;
            spin->grab().save(dir + QStringLiteral("w_spin.png"));
        }
        m_trainBtn->grab().save(dir + QStringLiteral("w_train.png"));
        m_tabLoss->grab().save(dir + QStringLiteral("w_tab.png"));
    }

protected:
    void closeEvent(QCloseEvent* e) override {
        if (g_bridge.isTraining) g_bridge.stopRequested = true; // let the worker break at the next epoch
        QMainWindow::closeEvent(e);
    }
    void dragEnterEvent(QDragEnterEvent* e) override {
        if (e->mimeData()->hasUrls()) e->acceptProposedAction();
    }
    void dropEvent(QDropEvent* e) override {
        for (const QUrl& url : e->mimeData()->urls()) {
            QString path = url.toLocalFile();
            if (path.endsWith(QStringLiteral(".csv"), Qt::CaseInsensitive)) {
                loadCsvFile(path);
                break;
            }
        }
    }

    void loadCsvFile(const QString& path) {
        m_csvPath = path.toStdString();
        m_dataset = DS_CSV;
        int cols = 0;
        {
            std::ifstream file(m_csvPath);
            if (file.is_open()) {
                std::string line;
                if (std::getline(file, line)) {
                    std::stringstream ss(line);
                    std::string cell;
                    while (std::getline(ss, cell, ',')) cols++;
                }
            }
        }
        m_targetCol->setMaximum(std::max(0, cols - 1));
        m_targetCol->setValue(-1);
        refreshConfigUi();
    }

private:
    // ---- experiment state (mirrors the widgets) ----
    ActiveDataset m_dataset = DS_XOR;
    std::string m_csvPath;
    ActiveLoss m_loss = LOSS_MSE;
    ActiveOptimizer m_opt = OPT_ADAM;
    std::string m_outputAct = "sigmoid";
    ViewMode m_view = VIEW_LOSS;
    double m_lr = 0.05;
    double m_momentum = 0.9, m_beta1 = 0.9, m_beta2 = 0.999, m_eps = 1e-8;
    int m_epochsTarget = 1500;
    int m_batchIdx = 0;
    int m_seed = 42;
    bool m_shuffle = true;
    bool m_splitShuffle = true;
    int m_trainPct = 80, m_valPct = 10;
    int m_normMode = 1;
    int m_numHidden = 2;
    int m_hiddenN[4] = {8, 8, 8, 8};
    ActiveActivation m_hiddenAct[4] = {ACT_TANH, ACT_RELU, ACT_RELU, ACT_RELU};
    UiState m_state = ST_IDLE;
    bool m_resultsShownFor = false;
    unsigned m_seenSeq = 0;

    // ---- dataset preview (recomputed when the data fingerprint changes) ----
    std::string m_previewFp;
    PreparedData m_preview;
    std::string m_previewErr;

    // ---- widgets ----
    QLabel* m_epochLbl = nullptr;
    QLabel* m_trainLossLbl = nullptr;
    QLabel* m_trainAccLbl = nullptr;
    QLabel* m_valAccLbl = nullptr;
    QLabel* m_testAccLbl = nullptr;
    QLabel* m_badge = nullptr;
    QWidget* m_configBox = nullptr;
    std::vector<QPushButton*> m_presetBtns;
    std::vector<QPushButton*> m_dsBtns;
    QLabel* m_dsInfoLbl = nullptr;
    QWidget* m_csvRow = nullptr;
    QLabel* m_csvPathLbl = nullptr;
    QSpinBox* m_targetCol = nullptr;
    QCheckBox* m_headerChk = nullptr;
    QLabel* m_layersLbl = nullptr;
    QLabel* m_archLbl = nullptr;
    struct LayerRow { QWidget* row; QLabel* name; QSpinBox* size; std::vector<QPushButton*> acts; };
    LayerRow m_layerRows[4];
    QComboBox* m_outActCombo = nullptr;
    QLabel* m_epochValLbl = nullptr;
    QSlider* m_epochSlider = nullptr;
    QSpinBox* m_epochSpin = nullptr;
    QComboBox* m_batchCombo = nullptr;
    QLabel* m_effBatchLbl = nullptr;
    QDoubleSpinBox* m_lrSpin = nullptr;
    QWidget* m_muRow = nullptr;
    QDoubleSpinBox* m_muSpin = nullptr;
    QWidget* m_adamRow = nullptr;
    QDoubleSpinBox* m_b1Spin = nullptr, *m_b2Spin = nullptr;
    QComboBox* m_epsCombo = nullptr;
    QSpinBox* m_seedSpin = nullptr;
    QCheckBox* m_shuffleChk = nullptr;
    QCheckBox* m_splitShuffleChk = nullptr;
    QSpinBox* m_trainPctSpin = nullptr, *m_valPctSpin = nullptr;
    QLabel* m_testPctLbl = nullptr;
    QComboBox* m_normCombo = nullptr;
    std::vector<QPushButton*> m_lossBtns;
    std::vector<QPushButton*> m_optBtns;
    QLabel* m_warnLbl = nullptr;
    QLabel* m_cfgLbl = nullptr;
    QLabel* m_summaryLbl = nullptr;
    QLabel* m_progressLbl = nullptr;
    QPushButton* m_trainBtn = nullptr;
    QPushButton* m_saveBtn = nullptr, *m_loadBtn = nullptr;
    QPushButton* m_tabLoss = nullptr, *m_tabAcc = nullptr, *m_tabBnd = nullptr, *m_tabNet = nullptr;
    // Exclusive checkable-button group: radio-button behavior for QPushButtons
    // without QButtonGroup's signal/slot machinery (no Q_OBJECT needed here).
    // Shared state keeps the click lambdas safe; groups are held as members
    // so the UI can re-sync checked states after presets/model loads.
    class ExclusiveButtonGroup {
    public:
        void addButton(QPushButton* b) {
            b->setCheckable(true);
            auto st = state_;
            QObject::connect(b, &QPushButton::clicked, [st, b]() {
                for (QPushButton* o : st->buttons) o->setChecked(o == b);
            });
            st->buttons.push_back(b);
            if (st->buttons.size() == 1) b->setChecked(true);
        }
        void checkOnly(int idx) const {
            for (int i = 0; i < (int)state_->buttons.size(); ++i)
                state_->buttons[(std::size_t)i]->setChecked(i == idx);
        }
    private:
        struct State { std::vector<QPushButton*> buttons; };
        std::shared_ptr<State> state_ = std::make_shared<State>();
    };
    ExclusiveButtonGroup m_dsGroup, m_lossGroup, m_optGroup, m_tabGroup;
    ExclusiveButtonGroup m_layerGroups[4];
    PlotWidget* m_plot = nullptr;
    BoundaryWidget* m_boundary = nullptr;
    NetWidget* m_netview = nullptr;
    QWidget* m_plotStack = nullptr;
    QLabel* m_summaryBar = nullptr;
    QTimer* m_timer = nullptr;

    QWidget* makeSection(const QString& text) {
        auto* w = new QWidget;
        auto* l = new QHBoxLayout(w);
        l->setContentsMargins(0, 0, 0, 0);
        l->setSpacing(8);
        auto* bar = new QFrame;
        bar->setFixedSize(4, 18);
        bar->setStyleSheet(QStringLiteral("background:#2563EB; border:none;"));
        auto* t = new QLabel(text);
        QFont f = t->font();
        f.setPointSize(10);
        f.setBold(true);
        t->setFont(f);
        t->setStyleSheet(QStringLiteral("color:#E8EAF0;"));
        l->addWidget(bar);
        l->addWidget(t, 1);
        return w;
    }

    QPushButton* makeBtn(const QString& text, int minWidth = 0) {
        auto* b = new QPushButton(text);
        b->setCursor(Qt::PointingHandCursor);
        // Explicit minimum height: QSS padding is not reliably reflected in
        // the layout sizeHint on every polish path, which previously squeezed
        // these buttons to ~17px and clipped their labels to dotted slivers.
        b->setMinimumHeight(30);
        if (minWidth > 0) b->setMinimumWidth(minWidth);
        return b;
    }

    QLabel* makeDim(const QString& text) {
        auto* l = new QLabel(text);
        l->setStyleSheet(QStringLiteral("color:#C9CDD8;"));
        return l;
    }

    // ---- config <-> ExperimentConfig ----
    ExperimentConfig currentConfig() const {
        ExperimentConfig c;
        c.dataset = kDsKeys[(int)m_dataset];
        c.csvPath = m_csvPath;
        c.targetCol = m_targetCol->value();
        c.header = m_headerChk->isChecked();
        c.normMode = m_normMode;
        c.seed = (unsigned)m_seed;
        c.shuffle = m_shuffle;
        c.splitShuffle = m_splitShuffle;
        c.trainFrac = m_trainPct / 100.0;
        c.valFrac = m_valPct / 100.0;
        for (int i = 0; i < m_numHidden; ++i) {
            c.hidden.push_back(m_hiddenN[i]);
            c.hiddenActs.push_back(actToName(m_hiddenAct[i]));
        }
        c.outputAct = m_outputAct;
        c.loss = kLossKeys[(int)m_loss];
        c.optimizer = kOptKeys[(int)m_opt];
        c.opt.learningRate = m_lr;
        c.opt.momentum = m_momentum;
        c.opt.beta1 = m_beta1;
        c.opt.beta2 = m_beta2;
        c.opt.epsilon = m_eps;
        c.epochs = m_epochsTarget;
        c.batchSize = kBatchVals[m_batchIdx];
        return c;
    }

    std::string fingerprint() const {
        return kDsKeys[(int)m_dataset] + std::string("|") + m_csvPath + "|" +
               std::to_string(m_targetCol->value()) + "|" + (m_headerChk->isChecked() ? "h" : "n") + "|" +
               std::to_string(m_trainPct) + "|" + std::to_string(m_valPct) + "|" +
               (m_splitShuffle ? "s" : "n") + std::to_string(m_normMode);
    }

    void refreshPreview() {
        std::string fp = fingerprint();
        if (fp == m_previewFp) return;
        m_previewFp = fp;
        m_previewErr.clear();
        try {
            m_preview = ExperimentController::prepare(currentConfig());
        } catch (const std::exception& e) {
            m_preview = PreparedData();
            m_previewErr = e.what();
        }
    }

    std::string archSummary() const {
        std::string inT = "?", outT = "?";
        if (m_previewErr.empty()) {
            inT = std::to_string(m_preview.inDim);
            outT = std::to_string(m_preview.outDim);
            if (m_dataset == DS_CSV && m_preview.inDim == 0) inT = "?";
        } else if (m_dataset == DS_IRIS) { inT = "4"; outT = "3"; }
        else if (m_dataset != DS_CSV) { inT = "2"; outT = "1"; }
        auto shortAct = [](const std::string& n) -> std::string {
            if (n == "sigmoid") return "Sig";
            if (n == "tanh") return "Tanh";
            if (n == "relu") return "ReLU";
            if (n == "leaky_relu") return "L-ReLU";
            if (n == "swish") return "Swish";
            return "Lin";
        };
        std::string s = inT;
        for (int i = 0; i < m_numHidden; ++i)
            s += " -> " + std::to_string(m_hiddenN[i]) + "(" + shortAct(actToName(m_hiddenAct[i])) + ")";
        s += " -> " + outT + "(" + shortAct(m_outputAct) + ")";
        return s;
    }

    void buildUi() {
        // ----- top bar: every metric is labeled with its source (§2.1) -----
        auto* top = new QFrame;
        top->setObjectName(QStringLiteral("topbar"));
        auto* topLay = new QHBoxLayout(top);
        topLay->setContentsMargins(24, 10, 24, 10);
        topLay->setSpacing(22);
        auto* titleLbl = new QLabel(QStringLiteral("MiniANN MVP - Network Workbench"));
        QFont titleFont = titleLbl->font();
        titleFont.setPointSize(15);
        titleLbl->setFont(titleFont);
        titleLbl->setStyleSheet(QStringLiteral("color:#F2F3F7;"));
        topLay->addWidget(titleLbl);
        topLay->addStretch(1);
        m_epochLbl = new QLabel;
        m_trainLossLbl = new QLabel;
        m_trainAccLbl = new QLabel;
        m_valAccLbl = new QLabel;
        m_testAccLbl = new QLabel;
        QFont statFont = m_epochLbl->font();
        statFont.setPointSize(10);
        m_epochLbl->setFont(statFont);
        m_trainLossLbl->setFont(statFont);
        m_trainAccLbl->setFont(statFont);
        m_valAccLbl->setFont(statFont);
        m_testAccLbl->setFont(statFont);
        m_epochLbl->setStyleSheet(QStringLiteral("color:#C9CDD8;"));
        m_trainLossLbl->setStyleSheet(QStringLiteral("color:#FACC15;"));
        m_trainAccLbl->setStyleSheet(QStringLiteral("color:#7FB3E8;"));
        m_valAccLbl->setStyleSheet(QStringLiteral("color:#F5A623;"));
        m_testAccLbl->setStyleSheet(QStringLiteral("color:#22C55E;"));
        topLay->addWidget(m_epochLbl);
        topLay->addWidget(m_trainLossLbl);
        topLay->addWidget(m_trainAccLbl);
        topLay->addWidget(m_valAccLbl);
        topLay->addWidget(m_testAccLbl);
        m_badge = new QLabel(QStringLiteral("IDLE"));
        m_badge->setAlignment(Qt::AlignCenter);
        m_badge->setFixedSize(96, 30);
        topLay->addWidget(m_badge);

        // ----- left config column -----
        auto* left = new QFrame;
        left->setObjectName(QStringLiteral("card"));
        left->setFixedWidth(496); // +16px compensates the scrollbar so the 5th activation button fits
        auto* outer = new QVBoxLayout(left);
        outer->setContentsMargins(20, 18, 20, 18);
        outer->setSpacing(10);
        m_configBox = new QWidget;
        auto* lv = new QVBoxLayout(m_configBox);
        lv->setContentsMargins(0, 0, 0, 0);
        lv->setSpacing(10);
        // The config column (~1150px of controls) is taller than short
        // windows: without scrolling, QVBoxLayout squeezes word-wrap labels
        // to 0px and overlaps rows, which painted as dotted/ghosted text.
        auto* cfgScroll = new QScrollArea;
        cfgScroll->setWidgetResizable(true);
        cfgScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        cfgScroll->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        cfgScroll->setFrameShape(QFrame::NoFrame);
        cfgScroll->setWidget(m_configBox);
        outer->addWidget(cfgScroll, 1);

        // presets (§41)
        auto* preRow = new QHBoxLayout;
        preRow->setSpacing(4);
        auto* preLbl = new QLabel(QStringLiteral("PRESETS"));
        preLbl->setStyleSheet(QStringLiteral("color:#8A90A0;"));
        preRow->addWidget(preLbl);
        const char* preNames[3] = {"XOR", "Iris", "Binary"};
        for (int i = 0; i < 3; ++i) {
            QPushButton* b = makeBtn(QString::fromLatin1(preNames[i]));
            b->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
            preRow->addWidget(b);
            m_presetBtns.push_back(b);
        }
        connect(m_presetBtns[0], &QPushButton::clicked, [this]() { applyPresetXor(true); });
        connect(m_presetBtns[1], &QPushButton::clicked, [this]() { applyPresetIris(); });
        connect(m_presetBtns[2], &QPushButton::clicked, [this]() { applyPresetBinary(); });
        lv->addLayout(preRow);

        // 1. dataset
        lv->addWidget(makeSection(QStringLiteral("1. DATASET")));
        auto* dsRow = new QHBoxLayout;
        dsRow->setSpacing(4);
        for (int i = 0; i < 5; ++i) {
            QPushButton* b = makeBtn(QString::fromLatin1(kDsNames[i]));
            b->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
            dsRow->addWidget(b);
            m_dsBtns.push_back(b);
        }
        for (QPushButton* b : m_dsBtns) m_dsGroup.addButton(b);
        for (int i = 0; i < 5; ++i) {
            connect(m_dsBtns[(std::size_t)i], &QPushButton::clicked, [this, i]() {
                m_dataset = ActiveDataset(i);
                configChanged();
            });
        }
        lv->addLayout(dsRow);

        // dataset info (§3.2)
        m_dsInfoLbl = new QLabel;
        m_dsInfoLbl->setObjectName(QStringLiteral("panel"));
        m_dsInfoLbl->setStyleSheet(QStringLiteral("color:#7FB3E8; padding:8px 12px;"));
        m_dsInfoLbl->setWordWrap(true);
        lv->addWidget(m_dsInfoLbl);

        // Dataset split (§8 of the split spec): test share is automatic.
        auto* splitRow = new QHBoxLayout;
        splitRow->setSpacing(6);
        splitRow->addWidget(makeDim(QStringLiteral("SPLIT")));
        m_trainPctSpin = new QSpinBox;
        m_trainPctSpin->setRange(1, 98);
        m_trainPctSpin->setValue(80);
        m_trainPctSpin->setFixedWidth(58);
        m_trainPctSpin->setSuffix(QStringLiteral("%"));
        connect(m_trainPctSpin, &QSpinBox::valueChanged, [this](int v) {
            m_trainPct = v;
            configChanged();
        });
        splitRow->addWidget(m_trainPctSpin);
        splitRow->addWidget(makeDim(QStringLiteral("train")));
        m_valPctSpin = new QSpinBox;
        m_valPctSpin->setRange(0, 98);
        m_valPctSpin->setValue(10);
        m_valPctSpin->setFixedWidth(58);
        m_valPctSpin->setSuffix(QStringLiteral("%"));
        connect(m_valPctSpin, &QSpinBox::valueChanged, [this](int v) {
            m_valPct = v;
            configChanged();
        });
        splitRow->addWidget(m_valPctSpin);
        splitRow->addWidget(makeDim(QStringLiteral("val")));
        m_testPctLbl = new QLabel;
        m_testPctLbl->setStyleSheet(QStringLiteral("color:#8A90A0;"));
        splitRow->addWidget(m_testPctLbl);
        splitRow->addStretch(1);
        lv->addLayout(splitRow);

        // CSV options row
        m_csvRow = new QWidget;
        auto* csvLay = new QVBoxLayout(m_csvRow);
        csvLay->setContentsMargins(0, 0, 0, 0);
        csvLay->setSpacing(4);
        auto* csvTop = new QHBoxLayout;
        csvTop->setSpacing(6);
        auto* tgtLbl = new QLabel(QStringLiteral("TARGET COL"));
        tgtLbl->setStyleSheet(QStringLiteral("color:#C9CDD8;"));
        csvTop->addWidget(tgtLbl);
        m_targetCol = new QSpinBox;
        m_targetCol->setRange(-1, 64);
        m_targetCol->setValue(-1);
        m_targetCol->setSpecialValueText(QStringLiteral("auto"));
        m_targetCol->setFixedWidth(70);
        connect(m_targetCol, &QSpinBox::valueChanged, [this]() { configChanged(); });
        csvTop->addWidget(m_targetCol);
        m_headerChk = new QCheckBox(QStringLiteral("header"));
        m_headerChk->setChecked(true);
        connect(m_headerChk, &QCheckBox::toggled, [this]() { configChanged(); });
        csvTop->addWidget(m_headerChk);
        csvTop->addStretch(1);
        csvLay->addLayout(csvTop);
        m_csvPathLbl = new QLabel;
        m_csvPathLbl->setStyleSheet(QStringLiteral("color:#8A90A0;"));
        m_csvPathLbl->setWordWrap(true);
        csvLay->addWidget(m_csvPathLbl);
        auto* browseBtn = makeBtn(QStringLiteral("Browse CSV file..."));
        browseBtn->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        connect(browseBtn, &QPushButton::clicked, [this]() {
            QString path = QFileDialog::getOpenFileName(this, QStringLiteral("Open CSV dataset"),
                                                        QString(), QStringLiteral("CSV files (*.csv);;All files (*)"));
            if (!path.isEmpty()) loadCsvFile(path);
        });
        csvLay->addWidget(browseBtn);
        lv->addWidget(m_csvRow);

        // 2. architecture (§5)
        lv->addWidget(makeSection(QStringLiteral("2. ARCHITECTURE")));
        auto* layRow = new QHBoxLayout;
        layRow->setSpacing(6);
        m_layersLbl = new QLabel;
        m_layersLbl->setStyleSheet(QStringLiteral("color:#C9CDD8;"));
        layRow->addWidget(m_layersLbl);
        auto* layMinus = makeBtn(QStringLiteral("-"), 30);
        auto* layPlus = makeBtn(QStringLiteral("+"), 30);
        connect(layMinus, &QPushButton::clicked, [this]() {
            m_numHidden = std::max(1, m_numHidden - 1);
            configChanged();
        });
        connect(layPlus, &QPushButton::clicked, [this]() {
            m_numHidden = std::min(4, m_numHidden + 1);
            configChanged();
        });
        layRow->addWidget(layMinus);
        layRow->addWidget(layPlus);
        layRow->addStretch(1);
        lv->addLayout(layRow);
        m_archLbl = new QLabel;
        m_archLbl->setStyleSheet(QStringLiteral("color:#7FB3E8;"));
        m_archLbl->setWordWrap(true);
        lv->addWidget(m_archLbl);
        for (int i = 0; i < 4; ++i) {
            auto* row = new QWidget;
            auto* rl = new QHBoxLayout(row);
            rl->setContentsMargins(0, 0, 0, 0);
            rl->setSpacing(4);
            auto* name = new QLabel;
            name->setStyleSheet(QStringLiteral("color:#C9CDD8;"));
            name->setFixedWidth(40);
            rl->addWidget(name);
            auto* sz = new QSpinBox;
            sz->setRange(1, 64);
            sz->setValue(8);
            sz->setFixedWidth(56);
            connect(sz, &QSpinBox::valueChanged, [this, i](int v) {
                m_hiddenN[i] = v;
                configChanged();
            });
            rl->addWidget(sz);
            std::vector<QPushButton*> acts;
            for (int a = 0; a < 5; ++a) {
                QPushButton* b = makeBtn(QString::fromLatin1(kActLabels[a]));
                b->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
                rl->addWidget(b);
                acts.push_back(b);
            }
            for (QPushButton* b : acts) m_layerGroups[i].addButton(b);
            for (int a = 0; a < 5; ++a) {
                connect(acts[(std::size_t)a], &QPushButton::clicked, [this, i, a]() {
                    m_hiddenAct[i] = ActiveActivation(a);
                    configChanged();
                });
            }
            m_layerRows[i] = {row, name, sz, acts};
            lv->addWidget(row);
        }
        auto* outRow = new QHBoxLayout;
        outRow->setSpacing(6);
        auto* outLbl = new QLabel(QStringLiteral("Output act:"));
        outLbl->setStyleSheet(QStringLiteral("color:#C9CDD8;"));
        outRow->addWidget(outLbl);
        m_outActCombo = new QComboBox;
        for (auto n : kOutActLabels) m_outActCombo->addItem(QString::fromLatin1(n));
        m_outActCombo->setCurrentIndex(0);
        connect(m_outActCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int i) {
            m_outputAct = kOutActKeys[std::max(0, std::min(5, i))];
            configChanged();
        });
        outRow->addWidget(m_outActCombo, 1);
        lv->addLayout(outRow);

        // 3. training (§9)
        lv->addWidget(makeSection(QStringLiteral("3. TRAINING")));
        auto* epTop = new QHBoxLayout;
        epTop->setSpacing(8);
        m_epochValLbl = new QLabel;
        m_epochValLbl->setStyleSheet(QStringLiteral("color:#FACC15;"));
        QFont epFont = m_epochValLbl->font();
        epFont.setPointSize(10);
        epFont.setBold(true);
        m_epochValLbl->setFont(epFont);
        epTop->addWidget(m_epochValLbl);
        auto* epLab = new QLabel(QStringLiteral("EPOCHS"));
        epLab->setStyleSheet(QStringLiteral("color:#8A90A0;"));
        epTop->addWidget(epLab);
        epTop->addStretch(1);
        lv->addLayout(epTop);
        auto* epRow = new QHBoxLayout;
        epRow->setSpacing(6);
        m_epochSlider = new QSlider(Qt::Horizontal);
        m_epochSlider->setRange(100, 10000);
        m_epochSlider->setSingleStep(50);
        m_epochSlider->setPageStep(500);
        m_epochSlider->setValue(1500);
        connect(m_epochSlider, &QSlider::valueChanged, [this](int v) {
            v = (v / 50) * 50;
            if (v != m_epochsTarget) {
                m_epochsTarget = v;
                configChanged();
            }
        });
        epRow->addWidget(m_epochSlider, 1);
        m_epochSpin = new QSpinBox;
        m_epochSpin->setRange(1, 200000);
        m_epochSpin->setSingleStep(50);
        m_epochSpin->setValue(1500);
        m_epochSpin->setFixedWidth(84);
        connect(m_epochSpin, &QSpinBox::valueChanged, [this](int v) {
            if (v != m_epochsTarget) {
                m_epochsTarget = v;
                configChanged();
            }
        });
        epRow->addWidget(m_epochSpin);
        auto* epMinus = makeBtn(QStringLiteral("-100"), 52);
        auto* epPlus = makeBtn(QStringLiteral("+100"), 52);
        connect(epMinus, &QPushButton::clicked, [this]() {
            m_epochsTarget = std::max(1, m_epochsTarget - 100);
            configChanged();
        });
        connect(epPlus, &QPushButton::clicked, [this]() {
            m_epochsTarget = std::min(200000, m_epochsTarget + 100);
            configChanged();
        });
        epRow->addWidget(epMinus);
        epRow->addWidget(epPlus);
        lv->addLayout(epRow);

        auto* batchRow = new QHBoxLayout;
        batchRow->setSpacing(8);
        batchRow->addWidget(makeDim(QStringLiteral("BATCH SIZE")));
        m_batchCombo = new QComboBox;
        for (auto n : kBatchLabels) m_batchCombo->addItem(QString::fromLatin1(n));
        m_batchCombo->setCurrentIndex(0);
        m_batchCombo->setFixedWidth(80);
        connect(m_batchCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int i) {
            m_batchIdx = std::max(0, std::min(6, i));
            configChanged();
        });
        batchRow->addWidget(m_batchCombo);
        m_effBatchLbl = new QLabel;
        m_effBatchLbl->setStyleSheet(QStringLiteral("color:#8A90A0;"));
        batchRow->addWidget(m_effBatchLbl);
        batchRow->addStretch(1);
        lv->addLayout(batchRow);

        auto* lrRow = new QHBoxLayout;
        lrRow->setSpacing(8);
        lrRow->addWidget(makeDim(QStringLiteral("LEARNING RATE")));
        m_lrSpin = new QDoubleSpinBox;
        m_lrSpin->setRange(0.0001, 1.0);
        m_lrSpin->setDecimals(4);
        m_lrSpin->setSingleStep(0.005);
        m_lrSpin->setValue(0.05);
        m_lrSpin->setFixedWidth(90);
        connect(m_lrSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [this](double v) {
            m_lr = v;
            configChanged();
        });
        lrRow->addWidget(m_lrSpin);
        lrRow->addWidget(makeDim(QStringLiteral("SEED")));
        m_seedSpin = new QSpinBox;
        m_seedSpin->setRange(0, 999999);
        m_seedSpin->setValue(42);
        m_seedSpin->setFixedWidth(80);
        connect(m_seedSpin, &QSpinBox::valueChanged, [this](int v) {
            m_seed = v;
            configChanged();
        });
        lrRow->addWidget(m_seedSpin);
        lrRow->addWidget(makeDim(QStringLiteral("SHUFFLE")));
        m_shuffleChk = new QCheckBox(QStringLiteral("epoch"));
        m_shuffleChk->setChecked(true);
        connect(m_shuffleChk, &QCheckBox::toggled, [this](bool b) {
            m_shuffle = b;
            configChanged();
        });
        lrRow->addWidget(m_shuffleChk);
        m_splitShuffleChk = new QCheckBox(QStringLiteral("split"));
        m_splitShuffleChk->setChecked(true);
        m_splitShuffleChk->setToolTip(QStringLiteral("Shuffle before the train/val/test split"));
        connect(m_splitShuffleChk, &QCheckBox::toggled, [this](bool b) {
            m_splitShuffle = b;
            configChanged();
        });
        lrRow->addWidget(m_splitShuffleChk);
        lrRow->addStretch(1);
        lv->addLayout(lrRow);

        auto* normRow = new QHBoxLayout;
        normRow->setSpacing(8);
        normRow->addWidget(makeDim(QStringLiteral("NORMALIZE")));
        m_normCombo = new QComboBox;
        m_normCombo->addItems({QStringLiteral("None"), QStringLiteral("MaxAbs"), QStringLiteral("MinMax")});
        m_normCombo->setCurrentIndex(1);
        connect(m_normCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int i) {
            m_normMode = i;
            configChanged();
        });
        normRow->addWidget(m_normCombo);
        normRow->addStretch(1);
        lv->addLayout(normRow);

        // 4. loss & optimizer (§7, §8)
        lv->addWidget(makeSection(QStringLiteral("4. LOSS & OPTIMIZER")));
        auto* lossRow = new QHBoxLayout;
        lossRow->setSpacing(4);
        for (int i = 0; i < 3; ++i) {
            QPushButton* b = makeBtn(QString::fromLatin1(kLossLabels[i]));
            b->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
            lossRow->addWidget(b);
            m_lossBtns.push_back(b);
        }
        for (QPushButton* b : m_lossBtns) m_lossGroup.addButton(b);
        for (int i = 0; i < 3; ++i) {
            connect(m_lossBtns[(std::size_t)i], &QPushButton::clicked, [this, i]() {
                m_loss = ActiveLoss(i);
                configChanged();
            });
        }
        lv->addLayout(lossRow);
        auto* optRow = new QHBoxLayout;
        optRow->setSpacing(4);
        for (int i = 0; i < 3; ++i) {
            QPushButton* b = makeBtn(QString::fromLatin1(kOptLabels[i]));
            b->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
            optRow->addWidget(b);
            m_optBtns.push_back(b);
        }
        for (QPushButton* b : m_optBtns) m_optGroup.addButton(b);
        for (int i = 0; i < 3; ++i) {
            connect(m_optBtns[(std::size_t)i], &QPushButton::clicked, [this, i]() {
                m_opt = ActiveOptimizer(i);
                applyOptimizerDefaults();
                configChanged();
            });
        }
        lv->addLayout(optRow);

        m_muRow = new QWidget;
        auto* muLay = new QHBoxLayout(m_muRow);
        muLay->setContentsMargins(0, 0, 0, 0);
        muLay->setSpacing(8);
        muLay->addWidget(makeDim(QStringLiteral("MOMENTUM")));
        m_muSpin = new QDoubleSpinBox;
        m_muSpin->setRange(0.0, 0.999);
        m_muSpin->setDecimals(3);
        m_muSpin->setSingleStep(0.05);
        m_muSpin->setValue(0.9);
        m_muSpin->setFixedWidth(80);
        connect(m_muSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [this](double v) {
            m_momentum = v;
            configChanged();
        });
        muLay->addWidget(m_muSpin);
        muLay->addStretch(1);
        lv->addWidget(m_muRow);

        m_adamRow = new QWidget;
        auto* adLay = new QHBoxLayout(m_adamRow);
        adLay->setContentsMargins(0, 0, 0, 0);
        adLay->setSpacing(6);
        adLay->addWidget(makeDim(QStringLiteral("B1")));
        m_b1Spin = new QDoubleSpinBox;
        m_b1Spin->setRange(0.5, 0.9999);
        m_b1Spin->setDecimals(4);
        m_b1Spin->setValue(0.9);
        m_b1Spin->setFixedWidth(76);
        connect(m_b1Spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [this](double v) {
            m_beta1 = v;
            configChanged();
        });
        adLay->addWidget(m_b1Spin);
        adLay->addWidget(makeDim(QStringLiteral("B2")));
        m_b2Spin = new QDoubleSpinBox;
        m_b2Spin->setRange(0.9, 0.99999);
        m_b2Spin->setDecimals(5);
        m_b2Spin->setValue(0.999);
        m_b2Spin->setFixedWidth(82);
        connect(m_b2Spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [this](double v) {
            m_beta2 = v;
            configChanged();
        });
        adLay->addWidget(m_b2Spin);
        adLay->addWidget(makeDim(QStringLiteral("EPS")));
        m_epsCombo = new QComboBox;
        m_epsCombo->addItems({QStringLiteral("1e-6"), QStringLiteral("1e-7"),
                              QStringLiteral("1e-8"), QStringLiteral("1e-9")});
        m_epsCombo->setCurrentIndex(2);
        connect(m_epsCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int i) {
            m_eps = std::pow(10.0, -6 - i);
            configChanged();
        });
        adLay->addWidget(m_epsCombo);
        adLay->addStretch(1);
        lv->addWidget(m_adamRow);

        m_warnLbl = new QLabel;
        m_warnLbl->setWordWrap(true);
        m_warnLbl->setStyleSheet(QStringLiteral("color:#F5A623;"));
        lv->addWidget(m_warnLbl);

        // 5. run status (§42 summary lives here)
        lv->addWidget(makeSection(QStringLiteral("5. RUN STATUS")));
        auto* panel = new QFrame;
        panel->setObjectName(QStringLiteral("panel"));
        auto* pv = new QVBoxLayout(panel);
        pv->setContentsMargins(12, 10, 12, 10);
        pv->setSpacing(4);
        m_cfgLbl = new QLabel;
        m_cfgLbl->setStyleSheet(QStringLiteral("color:#C9CDD8;"));
        m_summaryLbl = new QLabel;
        m_summaryLbl->setStyleSheet(QStringLiteral("color:#FACC15;"));
        m_progressLbl = new QLabel;
        m_progressLbl->setStyleSheet(QStringLiteral("color:#22C55E;"));
        for (QLabel* l : {m_cfgLbl, m_summaryLbl, m_progressLbl}) {
            l->setWordWrap(true);
            pv->addWidget(l);
        }
        lv->addWidget(panel);

        // (No dead stretch here: the scroll area above takes all extra space,
        // keeping SAVE/LOAD + START docked at the bottom.)
        auto* modelRow = new QHBoxLayout;
        modelRow->setSpacing(6);
        m_saveBtn = makeBtn(QStringLiteral("SAVE MODEL"));
        m_loadBtn = makeBtn(QStringLiteral("LOAD MODEL"));
        m_saveBtn->setEnabled(false);
        connect(m_saveBtn, &QPushButton::clicked, [this]() { onSaveModel(); });
        connect(m_loadBtn, &QPushButton::clicked, [this]() { onLoadModel(); });
        modelRow->addWidget(m_saveBtn, 1);
        modelRow->addWidget(m_loadBtn, 1);
        outer->addLayout(modelRow);
        m_trainBtn = new QPushButton(QStringLiteral("START TRAINING"));
        m_trainBtn->setObjectName(QStringLiteral("trainBtn"));
        m_trainBtn->setCursor(Qt::PointingHandCursor);
        m_trainBtn->setMinimumHeight(46);
        QFont trainFont = m_trainBtn->font();
        trainFont.setPointSize(11);
        trainFont.setBold(true);
        m_trainBtn->setFont(trainFont);
        connect(m_trainBtn, &QPushButton::clicked, [this]() { onTrainClicked(); });
        outer->addWidget(m_trainBtn);

        // ----- right column -----
        auto* right = new QFrame;
        right->setObjectName(QStringLiteral("card"));
        auto* rv = new QVBoxLayout(right);
        rv->setContentsMargins(20, 14, 20, 14);
        rv->setSpacing(10);
        auto* tabRow = new QHBoxLayout;
        tabRow->setSpacing(8);
        m_tabLoss = makeBtn(QStringLiteral("Loss Curves"), 130);
        m_tabAcc = makeBtn(QStringLiteral("Accuracy (%)"), 140);
        m_tabBnd = makeBtn(QStringLiteral("Boundary"), 110);
        m_tabNet = makeBtn(QStringLiteral("Network"), 110);
        m_tabGroup.addButton(m_tabLoss);
        m_tabGroup.addButton(m_tabAcc);
        m_tabGroup.addButton(m_tabBnd);
        m_tabGroup.addButton(m_tabNet);
        m_tabGroup.checkOnly(0);
        connect(m_tabLoss, &QPushButton::clicked, [this]() { m_view = VIEW_LOSS; syncTabs(); });
        connect(m_tabAcc, &QPushButton::clicked, [this]() { m_view = VIEW_ACC; syncTabs(); });
        connect(m_tabBnd, &QPushButton::clicked, [this]() { m_view = VIEW_BOUNDARY; syncTabs(); });
        connect(m_tabNet, &QPushButton::clicked, [this]() { m_view = VIEW_NETWORK; syncTabs(); });
        tabRow->addWidget(m_tabLoss);
        tabRow->addWidget(m_tabAcc);
        tabRow->addWidget(m_tabBnd);
        tabRow->addWidget(m_tabNet);
        tabRow->addStretch(1);
        rv->addLayout(tabRow);
        m_plotStack = new QWidget;
        auto* stackLay = new QVBoxLayout(m_plotStack);
        stackLay->setContentsMargins(0, 0, 0, 0);
        m_plot = new PlotWidget;
        m_plot->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        m_boundary = new BoundaryWidget;
        m_boundary->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        m_boundary->setVisible(false);
        m_netview = new NetWidget;
        m_netview->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        m_netview->setVisible(false);
        stackLay->addWidget(m_plot);
        stackLay->addWidget(m_boundary);
        stackLay->addWidget(m_netview);
        rv->addWidget(m_plotStack, 1);
        m_summaryBar = new QLabel(QStringLiteral("Ready to train."));
        m_summaryBar->setObjectName(QStringLiteral("panel"));
        m_summaryBar->setStyleSheet(QStringLiteral("color:#FACC15; padding:8px 12px;"));
        m_summaryBar->setWordWrap(true);
        rv->addWidget(m_summaryBar);

        // ----- central layout -----
        auto* central = new QWidget;
        auto* ch = new QHBoxLayout(central);
        ch->setContentsMargins(24, 12, 24, 12);
        ch->setSpacing(20);
        ch->addWidget(left);
        ch->addWidget(right, 1);

        auto* root = new QWidget;
        auto* rootLay = new QVBoxLayout(root);
        rootLay->setContentsMargins(0, 0, 0, 0);
        rootLay->setSpacing(0);
        rootLay->addWidget(top);
        rootLay->addWidget(central, 1);
        setCentralWidget(root);

        auto* foot = new QLabel(QStringLiteral("[F11] Maximize  |  Configure the experiment, check the dataset panel, then START TRAINING"));
        foot->setAlignment(Qt::AlignCenter);
        foot->setStyleSheet(QStringLiteral("color:#8A90A0;"));
        statusBar()->addWidget(foot, 1);
    }

    void applyTheme() {
        setStyleSheet(QStringLiteral(
            "QMainWindow, QWidget#qt_top { background:#0C0D14; }"
            "QFrame#topbar { background:#151722; border:none; border-bottom:1px solid #232738; }"
            "QFrame#card { background:#13151F; border:1px solid #25293B; border-radius:2px; }"
            "QFrame#panel { background:#181A24; border:1px solid #2A2E40; border-radius:2px; }"
            "QLabel { color:#E8EAF0; }"
            "QPushButton { background:#1B1E2B; color:#E8EAF0; border:1px solid #2E3448; border-radius:2px; padding:7px 10px; }"
            "QPushButton:hover { border:1px solid #2563EB; }"
            "QPushButton:checked { background:#2563EB; border:1px solid #7FB3E8; }"
            "QPushButton:disabled { background:#14161F; color:#6A7080; border:1px solid #232738; }"
            "QPushButton#trainBtn { background:#16A34A; border:1px solid #22C55E; border-radius:3px; }"
            "QPushButton#trainBtn:hover { background:#18B456; }"
            "QPushButton#stopBtn { background:#7F1D1D; border:1px solid #EF4444; border-radius:3px; }"
            "QPushButton#stopBtn:hover { background:#991B1B; }"
            "QSlider::groove:horizontal { background:#2A2E40; height:8px; border-radius:2px; }"
            "QSlider::handle:horizontal { background:#E8EAF0; width:12px; margin:-7px 0; border-radius:2px; }"
            "QSlider::sub-page:horizontal { background:#5AA9E6; border-radius:2px; }"
            "QSpinBox, QComboBox, QDoubleSpinBox { background:#1B1E2B; color:#E8EAF0; border:1px solid #2E3448; border-radius:2px; padding:4px; }"
            "QComboBox QAbstractItemView { background:#1B1E2B; color:#E8EAF0; selection-background-color:#2563EB; }"
            "QCheckBox { color:#C9CDD8; }"
            "QStatusBar { background:#11131C; color:#8A90A0; border-top:1px solid #232738; }"
            "QToolTip { background:#000000; color:#E8EAF0; border:1px solid #3A3F52; }"
            "QTableWidget { background:#181A24; color:#E8EAF0; gridline-color:#2A2E40; border:1px solid #2A2E40; }"
            "QScrollArea { background:transparent; border:none; }"
            "QScrollBar:vertical { background:transparent; width:12px; margin:0; border:none; }"
            "QScrollBar::handle:vertical { background:#2A2E40; min-height:30px; border-radius:6px; }"
            "QScrollBar::handle:vertical:hover { background:#3A3F52; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; border:none; }"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:none; border:none; }"
            "QHeaderView::section { background:#1B1E2B; color:#C9CDD8; border:1px solid #2A2E40; padding:4px; }"
            "QDialog { background:#13151F; }"
        ));
    }

    // ---- config pipeline: preview -> validate -> READY/ERROR ----
    void configChanged() {
        if (g_bridge.isTraining) return; // config locked during training
        g_bridge.resetLive();
        m_resultsShownFor = false;
        refreshConfigUi();
    }

    void refreshConfigUi() {
        m_dsGroup.checkOnly((int)m_dataset);
        m_lossGroup.checkOnly((int)m_loss);
        m_optGroup.checkOnly((int)m_opt);
        // Preview FIRST: arch summary + info panel below consume it.
        refreshPreview();
        m_muRow->setVisible(m_opt == OPT_MOMENTUM);
        m_adamRow->setVisible(m_opt == OPT_ADAM);
        m_csvRow->setVisible(m_dataset == DS_CSV);
        m_csvPathLbl->setText(m_csvPath.empty() ? QStringLiteral("No file — drag and drop a .csv file onto the window.")
                                                : QString::fromStdString(m_csvPath));
        m_layersLbl->setText(QString::asprintf("LAYERS: %d", m_numHidden));
        m_archLbl->setText(QString::fromStdString(archSummary()));
        for (int i = 0; i < 4; ++i) {
            m_layerRows[i].row->setVisible(i < m_numHidden);
            m_layerRows[i].name->setText(QString::asprintf("H%d:%d", i + 1, m_hiddenN[i]));
            if (m_layerRows[i].size->value() != m_hiddenN[i]) {
                m_layerRows[i].size->blockSignals(true);
                m_layerRows[i].size->setValue(m_hiddenN[i]);
                m_layerRows[i].size->blockSignals(false);
            }
            for (int a = 0; a < 5; ++a)
                m_layerRows[i].acts[(std::size_t)a]->setChecked(m_hiddenAct[i] == ActiveActivation(a));
        }
        m_epochValLbl->setText(QString::number(m_epochsTarget));
        if (m_epochSlider->value() != m_epochsTarget &&
            m_epochsTarget >= 100 && m_epochsTarget <= 10000) {
            m_epochSlider->blockSignals(true);
            m_epochSlider->setValue(m_epochsTarget);
            m_epochSlider->blockSignals(false);
        }
        if (m_epochSpin->value() != m_epochsTarget) {
            m_epochSpin->blockSignals(true);
            m_epochSpin->setValue(m_epochsTarget);
            m_epochSpin->blockSignals(false);
        }
        if (m_trainPctSpin->value() != m_trainPct) {
            m_trainPctSpin->blockSignals(true);
            m_trainPctSpin->setValue(m_trainPct);
            m_trainPctSpin->blockSignals(false);
        }
        if (m_valPctSpin->value() != m_valPct) {
            m_valPctSpin->blockSignals(true);
            m_valPctSpin->setValue(m_valPct);
            m_valPctSpin->blockSignals(false);
        }
        int testPct = 100 - m_trainPct - m_valPct;
        m_testPctLbl->setText(QString::asprintf("= %d%% test", testPct));
        if (testPct < 1) {
            // Invalid split (§8): train + val consume the whole dataset.
            m_state = ST_ERROR;
            m_dsInfoLbl->setText(QString::asprintf(
                "Dataset split is invalid:\ntrain %d%% + val %d%% leaves %d%% for test.",
                m_trainPct, m_valPct, testPct));
            m_warnLbl->setStyleSheet(QStringLiteral("color:#EF4444;"));
            m_warnLbl->setText(QStringLiteral("Lower train/val so that test gets at least 1%."));
            m_cfgLbl->setText(QStringLiteral("Fix the dataset split to continue."));
            m_trainBtn->setEnabled(false);
            refreshLiveUi();
            return;
        }

        refreshPreview();
        if (!m_previewErr.empty()) {
            m_state = ST_ERROR;
            m_dsInfoLbl->setText(QString::fromStdString(std::string("Dataset error:\n") + m_previewErr));
            m_warnLbl->setText(QString());
            m_cfgLbl->setText(QStringLiteral("Fix the dataset configuration to continue."));
            m_trainBtn->setEnabled(false);
            refreshLiveUi();
            return;
        }
        {
            const PreparedData& d = m_preview;
            QString info = QString::asprintf("Dataset: %s\nSamples: %llu | Features: %llu | Classes: ",
                                                     d.name.c_str(), (unsigned long long)d.samples,
                                                     (unsigned long long)d.features)
                + (d.discreteClasses ? QString::number((unsigned long long)d.classes) : QStringLiteral("—"))
                + QStringLiteral("\n");
            {
                QString tstat;
                if (d.targetDistinct <= 32)
                    tstat = QString::asprintf("%llu distinct values [%g..%g]",
                        (unsigned long long)d.targetDistinct, d.targetMin, d.targetMax);
                else
                    tstat = QString::asprintf("many distinct values [%g..%g]",
                        d.targetMin, d.targetMax);
                info += QString::asprintf("Target: col %d — ", d.targetColUsed) + tstat;
                if (!d.targetNote.empty())
                    info += QStringLiteral(" (") + QString::fromStdString(d.targetNote) + QStringLiteral(")");
                info += QStringLiteral("\n");
            }
            if (d.tiny)
                info += QString::asprintf("Train: %llu | Val: — | Test: — (full table: tiny dataset, no held-out test)\n",
                                          (unsigned long long)d.nTrain);
            else {
                int tp = (int)(100.0 * d.nTrain / (d.nTrain + d.nVal + d.nTest));
                int vp = (int)(100.0 * d.nVal / (d.nTrain + d.nVal + d.nTest));
                info += QString::asprintf("Train: %llu (%d%%) | Val: %llu (%d%%) | Test: %llu (%d%%)\n",
                                          (unsigned long long)d.nTrain, tp,
                                          (unsigned long long)d.nVal, vp,
                                          (unsigned long long)d.nTest, 100 - tp - vp);
            }
            info += QString::asprintf("Input dim: %llu | Output dim: %llu",
                                      (unsigned long long)d.inDim, (unsigned long long)d.outDim);
            m_dsInfoLbl->setText(info);

            std::size_t eff = (kBatchVals[m_batchIdx] == 0 || kBatchVals[m_batchIdx] > d.nTrain)
                ? d.nTrain : kBatchVals[m_batchIdx];
            m_effBatchLbl->setText(QString::asprintf("Effective: %llu", (unsigned long long)eff));

            std::string err, warn;
            ExperimentConfig cfg = currentConfig();
            if (!ExperimentController::validate(cfg, d, err, warn)) {
                m_state = ST_ERROR;
                m_warnLbl->setStyleSheet(QStringLiteral("color:#EF4444;"));
                m_warnLbl->setText(QString::fromStdString(err));
                m_trainBtn->setEnabled(false);
            } else {
                m_state = ST_READY;
                m_warnLbl->setStyleSheet(QStringLiteral("color:#F5A623;"));
                m_warnLbl->setText(QString::fromStdString(warn));
                m_trainBtn->setEnabled(true);
            }
            const char* normN = (m_normMode == 0) ? "none" : (m_normMode == 1) ? "maxabs" : "minmax";
            QString batchTxt = (kBatchVals[m_batchIdx] == 0) ? QStringLiteral("full")
                : QString::number((unsigned long long)kBatchVals[m_batchIdx]);
            m_cfgLbl->setText(QString::fromLatin1(kOptKeys[(int)m_opt]) + QString::asprintf(" | lr %g | ep %d | ", m_lr, m_epochsTarget)
                + QStringLiteral("batch ") + batchTxt
                + QString::asprintf(" (eff %llu) | seed %d%s%s | %s", (unsigned long long)eff, m_seed,
                                    m_splitShuffle ? "" : " nosplit-shuffle",
                                    m_shuffle ? "" : " noepoch-shuffle", normN));
        }
        refreshLiveUi();
    }

    void syncTabs() {
        m_tabGroup.checkOnly((int)m_view);
        m_plot->setVisible(m_view == VIEW_LOSS || m_view == VIEW_ACC);
        m_boundary->setVisible(m_view == VIEW_BOUNDARY);
        m_netview->setVisible(m_view == VIEW_NETWORK);
        refreshLiveUi();
    }

    static QString badgeStyle(const char* bg, const char* fg) {
        return QString::asprintf("background:%s; color:%s; border-radius:6px; font-weight:bold;", bg, fg);
    }

    // Per-tick change caches: refreshLiveUi runs at 10 Hz, and touching a
    // widget (setText/setStyleSheet/setObjectName/setEnabled) forces a
    // repolish + layout pass. Suppressing identical updates keeps the window
    // out of a perpetual re-layout storm that tears frames into dotted text.
    QString m_cEpoch, m_cLoss, m_cTrAcc, m_cValAcc, m_cTeAcc, m_cProg;
    QString m_cSum, m_cSumBar, m_cSumStyle, m_cTrainTxt, m_cTrainObj;
    int m_cBadge = -1, m_cCfgEn = -1, m_cSaveEn = -1, m_cTrainEn = -1;

    static void setOnce(QLabel* l, QString& cache, const QString& s) {
        if (s != cache) {
            l->setText(s);
            cache = s;
        }
    }
    void setSumStyle(const QString& st) {
        if (st != m_cSumStyle) {
            m_summaryLbl->setStyleSheet(st);
            m_cSumStyle = st;
        }
    }
    void setTrainBtn(const QString& txt, const char* obj, bool en) {
        if (txt != m_cTrainTxt) {
            m_trainBtn->setText(txt);
            m_cTrainTxt = txt;
        }
        QString o = QString::fromLatin1(obj);
        if (o != m_cTrainObj) {
            m_trainBtn->setObjectName(o);
            m_cTrainObj = o;
        }
        int e = en ? 1 : 0;
        if (e != m_cTrainEn) {
            m_trainBtn->setEnabled(en);
            m_cTrainEn = e;
        }
    }
    void setCfgEnabled(bool en) {
        int e = en ? 1 : 0;
        if (e != m_cCfgEn) {
            m_configBox->setEnabled(en);
            m_cCfgEn = e;
        }
    }
    void setSaveEnabled(bool en) {
        int e = en ? 1 : 0;
        if (e != m_cSaveEn) {
            m_saveBtn->setEnabled(en);
            m_cSaveEn = e;
        }
    }

    void setBadge(UiState s) {
        m_state = s;
        if ((int)s == m_cBadge) return;
        m_cBadge = (int)s;
        switch (s) {
        case ST_TRAINING:  m_badge->setText(QStringLiteral("TRAINING")); m_badge->setStyleSheet(badgeStyle("#14532D", "#22C55E")); break;
        case ST_COMPLETED: m_badge->setText(QStringLiteral("COMPLETE")); m_badge->setStyleSheet(badgeStyle("#134E4A", "#2DD4BF")); break;
        case ST_STOPPED:   m_badge->setText(QStringLiteral("STOPPED"));  m_badge->setStyleSheet(badgeStyle("#713F12", "#FBBF24")); break;
        case ST_ERROR:     m_badge->setText(QStringLiteral("ERROR"));    m_badge->setStyleSheet(badgeStyle("#7F1D1D", "#F87171")); break;
        case ST_READY:     m_badge->setText(QStringLiteral("READY"));    m_badge->setStyleSheet(badgeStyle("#1E3A8A", "#93C5FD")); break;
        default:           m_badge->setText(QStringLiteral("IDLE"));     m_badge->setStyleSheet(badgeStyle("#27272A", "#C9CDD8")); break;
        }
    }

    // ---- live refresh (10 Hz, GUI thread) ----
    void refreshLiveUi() {
        std::vector<float> trainLoss, valLoss, trainAcc, valAcc;
        int liveEpoch = 0;
        bool training = g_bridge.isTraining;
        std::string liveSummary;
        bool hasResult = false, completed = false, stopped = false;
        std::string lastError;
        double fTL = 0, fTA = 0, fVA = 0, fTeA = 0;
        bool fHasVal = false, fHasTest = false;
        unsigned seq = 0;
        {
            std::lock_guard<std::mutex> lock(g_bridge.mtx);
            trainLoss = g_bridge.trainLoss;
            valLoss = g_bridge.valLoss;
            trainAcc = g_bridge.trainAcc;
            valAcc = g_bridge.valAcc;
            liveEpoch = g_bridge.currentEpoch;
            liveSummary = g_bridge.summary;
            hasResult = g_bridge.hasResult;
            completed = g_bridge.lastCompleted;
            stopped = g_bridge.lastStopped;
            lastError = g_bridge.lastError;
            fTL = g_bridge.lastTrainLoss; fTA = g_bridge.lastTrainAcc;
            fVA = g_bridge.lastValAcc;
            fTeA = g_bridge.lastTestAcc;
            fHasVal = g_bridge.lastHasVal;
            fHasTest = g_bridge.lastHasTest;
            seq = g_bridge.runSeq;
        }

        const QString na = QStringLiteral("—");
        if (training) {
            setBadge(ST_TRAINING);
            setOnce(m_epochLbl, m_cEpoch, QString::asprintf("Epoch: %d / %d", liveEpoch, m_epochsTarget));
            setOnce(m_trainLossLbl, m_cLoss, trainLoss.empty() ? "Train Loss: —"
                : QString::asprintf("Train Loss: %.4f", trainLoss.back()));
            setOnce(m_trainAccLbl, m_cTrAcc, trainAcc.empty() ? "Train Acc: —"
                : QString::asprintf("Train Acc: %.1f%%", trainAcc.back() * 100.0f));
            setOnce(m_valAccLbl, m_cValAcc, valAcc.empty() ? "Val Acc: —"
                : QString::asprintf("Val Acc: %.1f%%", valAcc.back() * 100.0f));
            setOnce(m_testAccLbl, m_cTeAcc, QStringLiteral("Test Acc: —"));
            int pct = m_epochsTarget > 0 ? int(100.0f * liveEpoch / m_epochsTarget) : 0;
            setOnce(m_progressLbl, m_cProg, QString::asprintf("progress: %d%%", pct));
            setTrainBtn(QStringLiteral("STOP TRAINING"), "stopBtn", true);
            setCfgEnabled(false);
            setSumStyle(QStringLiteral("color:#FACC15;"));
            setOnce(m_summaryLbl, m_cSum, QStringLiteral("Training..."));
            setOnce(m_summaryBar, m_cSumBar, QStringLiteral("Training..."));
        } else if (!lastError.empty()) {
            setBadge(ST_ERROR);
            freezeHeader(na);
            setSumStyle(QStringLiteral("color:#F87171;"));
            setOnce(m_summaryLbl, m_cSum, QString::fromStdString(lastError));
            setOnce(m_summaryBar, m_cSumBar, QString::fromStdString(lastError));
            setTrainBtn(QStringLiteral("START TRAINING"), "trainBtn", m_previewErr.empty());
            setCfgEnabled(true);
            setSaveEnabled(hasResult && completed);
        } else if (hasResult) {
            setBadge(stopped ? ST_STOPPED : ST_COMPLETED);
            setOnce(m_epochLbl, m_cEpoch, QString::asprintf("Epoch: %d / %d", liveEpoch, m_epochsTarget));
            setOnce(m_trainLossLbl, m_cLoss, QString::asprintf("Train Loss: %.4f", fTL));
            setOnce(m_trainAccLbl, m_cTrAcc, QString::asprintf("Train Acc: %.1f%%", fTA * 100.0f));
            setOnce(m_valAccLbl, m_cValAcc, fHasVal ? QString::asprintf("Val Acc: %.1f%%", fVA * 100.0f) : "Val Acc: —");
            setOnce(m_testAccLbl, m_cTeAcc, (fHasTest && hasResult)
                ? QString::asprintf("Test Acc: %.1f%%", fTeA * 100.0f) : "Test Acc: —");
            setSumStyle(QStringLiteral("color:#FACC15;"));
            setOnce(m_summaryBar, m_cSumBar, QString::fromStdString(liveSummary));
            setOnce(m_summaryLbl, m_cSum, QString::fromStdString(liveSummary));
            setOnce(m_progressLbl, m_cProg, QString());
            setTrainBtn(QStringLiteral("START TRAINING"), "trainBtn", true);
            setCfgEnabled(true);
            setSaveEnabled(completed);
            if (completed && !m_resultsShownFor) {
                m_resultsShownFor = true;
                showResultsDialog();
            }
            if (seq != m_seenSeq) {
                m_seenSeq = seq;
                rebuildVizCaches();
            }
        } else {
            // Fresh / reconfigured: nothing evaluated yet — show "—", never 0% (§31).
            setBadge(m_state == ST_ERROR ? ST_ERROR : ST_READY);
            freezeHeader(na);
            setSumStyle(QStringLiteral("color:#FACC15;"));
            setOnce(m_summaryLbl, m_cSum, QString::fromStdString(liveSummary));
            setOnce(m_summaryBar, m_cSumBar, QString::fromStdString(liveSummary));
            setOnce(m_progressLbl, m_cProg, QString());
            setTrainBtn(QStringLiteral("START TRAINING"), "trainBtn", m_state != ST_ERROR);
            setCfgEnabled(true);
            setSaveEnabled(false);
        }

        m_plot->setCurves(m_view == VIEW_ACC ? trainAcc : trainLoss,
                          m_view == VIEW_ACC ? valAcc : valLoss,
                          m_view == VIEW_ACC, !valLoss.empty() || !valAcc.empty(), liveEpoch);
    }

    void freezeHeader(const QString& na) {
        setOnce(m_epochLbl, m_cEpoch, QString::asprintf("Epoch: 0 / %d", m_epochsTarget));
        setOnce(m_trainLossLbl, m_cLoss, QStringLiteral("Train Loss: ") + na);
        setOnce(m_trainAccLbl, m_cTrAcc, QStringLiteral("Train Acc: ") + na);
        setOnce(m_valAccLbl, m_cValAcc, QStringLiteral("Val Acc: ") + na);
        setOnce(m_testAccLbl, m_cTeAcc, QStringLiteral("Test Acc: ") + na);
    }

    // ---- training control ----
    void onTrainClicked() {
        if (g_bridge.isTraining) {
            g_bridge.stopRequested = true; // STOP button (§10)
            return;
        }
        if (m_state == ST_ERROR) return;
        ExperimentConfig cfg = currentConfig();
        // Fresh curves + cleared result: the new run must not append to stale data.
        g_bridge.resetLive();
        m_resultsShownFor = false;
        m_seenSeq = g_bridge.runSeq;
        g_bridge.stopRequested = false;
        g_bridge.isTraining = true;
        if (g_trainThread && g_trainThread->joinable()) g_trainThread->join();
        g_trainThread = std::make_unique<std::thread>([cfg]() {
            ExperimentResult res = ExperimentController::run(cfg, &g_bridge, &g_bridge.stopRequested);
            std::lock_guard<std::mutex> lock(g_bridge.mtx);
            if (!res.error.empty()) {
                g_bridge.lastError = std::string("Error: ") + res.error;
            } else {
                if (res.hasTest) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                                  "Train loss %.4f | Train acc %.1f%% | Test acc %.1f%% (%.2fs)%s",
                                  res.trainLoss, res.trainAcc * 100.0, res.testAcc * 100.0,
                                  res.seconds, res.stopped ? " — stopped" : "");
                    g_bridge.summary = buf;
                } else {
                    // No held-out test set exists: report train metrics only,
                    // never relabel them as test metrics.
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                                  "Train loss %.4f | Train acc %.1f%% (%.2fs)%s — no held-out test set",
                                  res.trainLoss, res.trainAcc * 100.0,
                                  res.seconds, res.stopped ? ", stopped" : "");
                    g_bridge.summary = buf;
                }
                g_bridge.hasResult = true;
                g_bridge.lastCompleted = !res.stopped;
                g_bridge.lastStopped = res.stopped;
                g_bridge.lastTrainLoss = res.trainLoss;
                g_bridge.lastTrainAcc = res.trainAcc;
                g_bridge.lastValLoss = res.valLoss;
                g_bridge.lastValAcc = res.valAcc;
                g_bridge.lastTestLoss = res.testLoss;
                g_bridge.lastTestAcc = res.testAcc;
                g_bridge.lastHasVal = res.hasVal;
                g_bridge.lastHasTest = res.hasTest;
                g_bridge.lastWarning = res.warning;
                g_bridge.lastConfusion = res.confusion;
                g_bridge.lastNumClasses = res.numClasses;
                g_bridge.lastHasConfusion = res.hasConfusion;
                g_bridge.lastConfusionOnTrain = res.confusionOnTrain;
                if (res.hasTest) {
                    char splitBuf[128];
                    std::snprintf(splitBuf, sizeof(splitBuf), "Train %llu | Val %llu | Test %llu",
                                  (unsigned long long)res.data.nTrain, (unsigned long long)res.data.nVal,
                                  (unsigned long long)res.data.nTest);
                    g_bridge.lastSplitTxt = splitBuf;
                } else {
                    g_bridge.lastSplitTxt = "Train " + std::to_string(res.data.nTrain) +
                        " (full table — no held-out val/test)";
                }
                g_bridge.lastArch = describeNet(res.net);
                g_bridge.lastOptDesc = cfg.optimizer + " lr=" + std::to_string(cfg.opt.learningRate);
                std::size_t effB = (cfg.batchSize == 0 || cfg.batchSize > res.data.nTrain)
                    ? res.data.nTrain : cfg.batchSize;
                g_bridge.lastBatchTxt = (cfg.batchSize == 0 ? std::string("full")
                    : std::to_string(cfg.batchSize)) + " (eff " + std::to_string(effB) + ")";
                g_bridge.lastSeed = (int)cfg.seed;
                g_bridge.lastDsDesc = res.data.name + " " + std::to_string(res.data.samples) + " samples";
                g_bridge.lastLoss = cfg.loss;
                if (res.hasNet) {
                    g_bridge.lastNet = std::make_shared<NeuralNetwork>(res.net);
                    g_bridge.lastTrain = res.data.train;
                    g_bridge.lastTest = res.data.test;
                    g_bridge.lastInDim = res.data.inDim;
                    g_bridge.lastOutDim = res.data.outDim;
                    g_bridge.lastEpochsRun = (int)res.history.trainLoss.size();
                    g_bridge.lastEpochsTarget = cfg.epochs;
                    g_bridge.lastTiny = res.data.tiny;
                }
                g_bridge.runSeq++;
            }
            g_bridge.isTraining = false;
        });
    }

    static std::string describeNet(const NeuralNetwork& net) {
        if (net.numLayers() == 0) return "?";
        std::string s = std::to_string(net.layers()[0].inputSize());
        for (const auto& l : net.layers()) {
            std::string an = l.neurons().empty() ? "?" : l.neurons()[0].activation().name();
            s += " -> " + std::to_string(l.size()) + "(" + an + ")";
        }
        return s;
    }

    // ---- visualization caches (rebuilt once per completed run) ----
    void rebuildVizCaches() {
        std::shared_ptr<NeuralNetwork> net;
        Dataset tr, te;
        std::size_t inDim = 0;
        {
            std::lock_guard<std::mutex> lock(g_bridge.mtx);
            net = g_bridge.lastNet;
            tr = g_bridge.lastTrain;
            te = g_bridge.lastTest;
            inDim = g_bridge.lastInDim;
        }
        // Boundary (2D inputs only).
        if (net && inDim == 2 && tr.size() > 0) {
            double x0 = 1e300, x1 = -1e300, y0 = 1e300, y1 = -1e300;
            auto eat = [&](const Dataset& d) {
                for (std::size_t i = 0; i < d.size(); ++i) {
                    x0 = std::min(x0, d.input(i)[0]); x1 = std::max(x1, d.input(i)[0]);
                    y0 = std::min(y0, d.input(i)[1]); y1 = std::max(y1, d.input(i)[1]);
                }
            };
            eat(tr); eat(te);
            double dx = (x1 - x0) == 0 ? 1.0 : (x1 - x0), dy = (y1 - y0) == 0 ? 1.0 : (y1 - y0);
            x0 -= dx * 0.15; x1 += dx * 0.15; y0 -= dy * 0.15; y1 += dy * 0.15;
            const int N = 90;
            QImage grid(N, N, QImage::Format_RGB32);
            bool multi = net->layers().back().size() > 1;
            for (int gy = 0; gy < N; ++gy) {
                for (int gx = 0; gx < N; ++gx) {
                    double px = x0 + (gx + 0.5) / N * (x1 - x0);
                    double py = y1 - (gy + 0.5) / N * (y1 - y0);
                    Vector out = net->predict({px, py});
                    QColor c;
                    if (multi) {
                        std::size_t b = 0;
                        for (std::size_t k = 1; k < out.size(); ++k)
                            if (out[k] > out[b]) b = k;
                        c = QColor(kClassColors[b % 6]);
                    } else {
                        double v = std::max(0.0, std::min(1.0, out[0]));
                        QColor c0(kClassColors[0]), c1(kClassColors[1]);
                        c = QColor(int(c0.red() + (c1.red() - c0.red()) * v),
                                   int(c0.green() + (c1.green() - c0.green()) * v),
                                   int(c0.blue() + (c1.blue() - c0.blue()) * v));
                    }
                    c = c.darker(320);
                    grid.setPixel(gx, gy, c.rgb());
                }
            }
            auto toCls = [&](const Vector& t) {
                if (t.size() == 1) return t[0] >= 0.5 ? 1 : 0;
                std::size_t b = 0;
                for (std::size_t k = 1; k < t.size(); ++k)
                    if (t[k] > t[b]) b = k;
                return (int)b;
            };
            std::vector<QPointF> trP, teP;
            std::vector<int> trC, teC;
            for (std::size_t i = 0; i < tr.size(); ++i) {
                trP.push_back(QPointF(tr.input(i)[0], tr.input(i)[1]));
                trC.push_back(toCls(tr.target(i)));
            }
            for (std::size_t i = 0; i < te.size(); ++i) {
                teP.push_back(QPointF(te.input(i)[0], te.input(i)[1]));
                teC.push_back(toCls(te.target(i)));
            }
            m_boundary->setData(std::move(grid), std::move(trP), std::move(trC),
                                std::move(teP), std::move(teC), x0, x1, y0, y1, true);
        } else {
            m_boundary->setData(QImage(), {}, {}, {}, {}, 0, 1, 0, 1, false);
        }
        // Network diagram.
        ArchDesc ad;
        if (net && net->numLayers() > 0) {
            ad.valid = true;
            ad.sizes.push_back(net->layers()[0].inputSize());
            ad.names.push_back(QStringLiteral("Input"));
            for (std::size_t l = 0; l < net->numLayers(); ++l) {
                const auto& layer = net->layers()[l];
                ad.sizes.push_back(layer.size());
                QString an = layer.neurons().empty()
                    ? QStringLiteral("?")
                    : QString::fromStdString(layer.neurons()[0].activation().name());
                bool last = (l + 1 == net->numLayers());
                ad.names.push_back((last ? QStringLiteral("Output (") : QStringLiteral("Hidden ")) +
                                   (last ? an + QStringLiteral(")")
                                         : QString::number((unsigned long long)l + 1) +
                                           QStringLiteral(" (") + an + QStringLiteral(")")));
            }
        }
        m_netview->setArch(std::move(ad));
    }

    // ---- results dialog (§13) + model save/load (§27) ----
    void showResultsDialog() {
        std::lock_guard<std::mutex> lock(g_bridge.mtx);
        if (!g_bridge.hasResult || !g_bridge.lastCompleted) return;
        auto* dlg = new QDialog(this);
        dlg->setWindowTitle(QStringLiteral("Training Complete"));
        dlg->setAttribute(Qt::WA_DeleteOnClose);
        dlg->setMinimumWidth(460);
        auto* lay = new QVBoxLayout(dlg);
        auto* title = new QLabel(QStringLiteral("TRAINING COMPLETE"));
        QFont tf = title->font();
        tf.setPointSize(13);
        tf.setBold(true);
        title->setFont(tf);
        title->setStyleSheet(QStringLiteral("color:#2DD4BF;"));
        title->setAlignment(Qt::AlignCenter);
        lay->addWidget(title);
        auto* form = new QFormLayout;
        auto row = [&](const char* k, const QString& v) {
            auto* key = new QLabel(QString::fromLatin1(k));
            key->setStyleSheet(QStringLiteral("color:#8A90A0;"));
            auto* val = new QLabel(v);
            val->setWordWrap(true);
            form->addRow(key, val);
        };
        row("Dataset", QString::fromStdString(g_bridge.lastDsDesc));
        row("Split", QString::fromStdString(g_bridge.lastSplitTxt));
        row("Architecture", QString::fromStdString(g_bridge.lastArch));
        row("Optimizer", QString::fromStdString(g_bridge.lastOptDesc));
        row("Loss", QString::fromStdString(g_bridge.lastLoss));
        row("Batch Size", QString::fromStdString(g_bridge.lastBatchTxt));
        row("Seed", QString::number(g_bridge.lastSeed));
        row("Epochs", QString::asprintf("%d", g_bridge.lastEpochsRun));
        row("Train Loss", QString::asprintf("%.4f", g_bridge.lastTrainLoss));
        row("Train Acc", QString::asprintf("%.1f%%", g_bridge.lastTrainAcc * 100.0));
        if (g_bridge.lastHasVal) {
            row("Val Loss", QString::asprintf("%.4f", g_bridge.lastValLoss));
            row("Val Acc", QString::asprintf("%.1f%%", g_bridge.lastValAcc * 100.0));
        }
        if (g_bridge.lastHasTest) {
            row("Test Loss", QString::asprintf("%.4f", g_bridge.lastTestLoss));
            row("Test Acc", QString::asprintf("%.1f%%", g_bridge.lastTestAcc * 100.0));
        } else {
            row("Test Loss", QStringLiteral("n/a (no held-out test set)"));
            row("Test Acc", QStringLiteral("n/a (no held-out test set)"));
        }
        lay->addLayout(form);
        if (g_bridge.lastHasConfusion && g_bridge.lastNumClasses >= 2 && g_bridge.lastNumClasses <= 10) {
            auto* cmTitle = new QLabel(g_bridge.lastConfusionOnTrain
                ? QStringLiteral("Confusion matrix on the TRAIN set (no held-out test exists)")
                : QStringLiteral("Confusion matrix on the TEST set (rows = actual, cols = predicted)"));
            cmTitle->setStyleSheet(QStringLiteral("color:#8A90A0;"));
            lay->addWidget(cmTitle);
            std::size_t k = g_bridge.lastNumClasses;
            auto* table = new QTableWidget((int)k, (int)k);
            table->setEditTriggers(QAbstractItemView::NoEditTriggers);
            table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
            table->verticalHeader()->setSectionResizeMode(QHeaderView::Stretch);
            for (std::size_t i = 0; i < k; ++i) {
                table->setHorizontalHeaderItem((int)i, new QTableWidgetItem(QString::number((unsigned long long)i)));
                table->setVerticalHeaderItem((int)i, new QTableWidgetItem(QString::number((unsigned long long)i)));
                for (std::size_t j = 0; j < k; ++j) {
                    auto* item = new QTableWidgetItem(
                        QString::number((unsigned long long)g_bridge.lastConfusion[i][j]));
                    item->setTextAlignment(Qt::AlignCenter);
                    if (i == j) {
                        item->setBackground(QColor(0x14, 0x53, 0x2D));
                        item->setForeground(QColor(0x22, 0xC5, 0x5E));
                    }
                    table->setItem((int)i, (int)j, item);
                }
            }
            table->setMinimumHeight(int(34 * k + 30));
            lay->addWidget(table);
        }
        auto* buttons = new QDialogButtonBox(QDialogButtonBox::Close);
        connect(buttons, &QDialogButtonBox::rejected, dlg, &QDialog::close);
        lay->addWidget(buttons);
        dlg->show();
    }

    void onSaveModel() {
        std::shared_ptr<NeuralNetwork> net;
        {
            std::lock_guard<std::mutex> lock(g_bridge.mtx);
            net = g_bridge.lastNet;
        }
        if (!net) return;
        QString path = QFileDialog::getSaveFileName(this, QStringLiteral("Save model"), QStringLiteral("model.miniann"),
                                                    QStringLiteral("MiniANN models (*.miniann);;All files (*)"));
        if (path.isEmpty()) return;
        try {
            ModelSerializer::save(*net, path.toStdString());
        } catch (const std::exception& e) {
            QMessageBox::warning(this, QStringLiteral("Save model"), QString::fromStdString(std::string("Could not save model:\n") + e.what()));
        }
    }

    void onLoadModel() {
        QString path = QFileDialog::getOpenFileName(this, QStringLiteral("Load model"),
                                                    QString(), QStringLiteral("MiniANN models (*.miniann);;All files (*)"));
        if (path.isEmpty()) return;
        try {
            std::mt19937 rng((unsigned)m_seed);
            NeuralNetwork net = ModelSerializer::load(path.toStdString(), rng);
            if (net.numLayers() < 2 || net.numLayers() > 5) {
                QMessageBox::warning(this, QStringLiteral("Load model"),
                                     QStringLiteral("Model has %1 layers; the editor supports 1-4 hidden layers.")
                                         .arg((unsigned long long)net.numLayers() - 1));
                return;
            }
            std::size_t nh = net.numLayers() - 1;
            m_numHidden = (int)nh;
            for (std::size_t l = 0; l < nh; ++l) {
                m_hiddenN[l] = (int)net.layers()[l].size();
                std::string an = net.layers()[l].neurons().empty()
                    ? "" : net.layers()[l].neurons()[0].activation().name();
                try {
                    m_hiddenAct[l] = actFromName(an);
                } catch (const std::exception&) {
                    QMessageBox::warning(this, QStringLiteral("Load model"),
                                         QString::fromStdString("Unsupported hidden activation '" + an + "'."));
                    return;
                }
            }
            std::string oan = net.layers().back().neurons().empty()
                ? "" : net.layers().back().neurons()[0].activation().name();
            bool known = false;
            for (int i = 0; i < 6; ++i)
                if (oan == kOutActKeys[i]) {
                    m_outputAct = oan;
                    m_outActCombo->setCurrentIndex(i);
                    known = true;
                }
            if (!known) {
                QMessageBox::warning(this, QStringLiteral("Load model"),
                                     QString::fromStdString("Unsupported output activation '" + oan + "'."));
                return;
            }
            {
                std::lock_guard<std::mutex> lock(g_bridge.mtx);
                g_bridge.lastNet = std::make_shared<NeuralNetwork>(net);
                g_bridge.lastInDim = net.layers()[0].inputSize();
                g_bridge.lastOutDim = net.layers().back().size();
            }
            ArchDesc ad;
            ad.valid = true;
            ad.sizes.push_back(net.layers()[0].inputSize());
            ad.names.push_back(QStringLiteral("Input"));
            for (std::size_t l = 0; l < net.numLayers(); ++l) {
                ad.sizes.push_back(net.layers()[l].size());
                QString an = QString::fromStdString(net.layers()[l].neurons()[0].activation().name());
                bool last = (l + 1 == net.numLayers());
                ad.names.push_back(last ? QStringLiteral("Output (") + an + QStringLiteral(")")
                                        : QStringLiteral("Hidden ") + QString::number((unsigned long long)l + 1) +
                                          QStringLiteral(" (") + an + QStringLiteral(")"));
            }
            m_netview->setArch(std::move(ad));
            pushUiFromState();
            configChanged();
            std::lock_guard<std::mutex> lock(g_bridge.mtx);
            g_bridge.summary = "Model loaded (" + describeNet(net) + ") — pick a matching dataset, then train or inspect Network.";
        } catch (const std::exception& e) {
            QMessageBox::warning(this, QStringLiteral("Load model"),
                                 QString::fromStdString(std::string("Could not load model:\n") + e.what()));
        }
    }

    void pushUiFromState() {
        m_dsGroup.checkOnly((int)m_dataset);
        m_lossGroup.checkOnly((int)m_loss);
        m_optGroup.checkOnly((int)m_opt);
        m_outActCombo->blockSignals(true);
        for (int i = 0; i < 6; ++i)
            if (m_outputAct == kOutActKeys[i]) m_outActCombo->setCurrentIndex(i);
        m_outActCombo->blockSignals(false);
        m_lrSpin->blockSignals(true);
        m_lrSpin->setValue(m_lr);
        m_lrSpin->blockSignals(false);
        m_muSpin->blockSignals(true);
        m_muSpin->setValue(m_momentum);
        m_muSpin->blockSignals(false);
        m_b1Spin->blockSignals(true);
        m_b1Spin->setValue(m_beta1);
        m_b1Spin->blockSignals(false);
        m_b2Spin->blockSignals(true);
        m_b2Spin->setValue(m_beta2);
        m_b2Spin->blockSignals(false);
        m_epochSpin->blockSignals(true);
        m_epochSpin->setValue(m_epochsTarget);
        m_epochSpin->blockSignals(false);
        m_epochSlider->blockSignals(true);
        m_epochSlider->setValue(std::max(100, std::min(10000, m_epochsTarget)));
        m_epochSlider->blockSignals(false);
        m_batchCombo->blockSignals(true);
        m_batchCombo->setCurrentIndex(m_batchIdx);
        m_batchCombo->blockSignals(false);
        m_seedSpin->blockSignals(true);
        m_seedSpin->setValue(m_seed);
        m_seedSpin->blockSignals(false);
        m_shuffleChk->blockSignals(true);
        m_shuffleChk->setChecked(m_shuffle);
        m_shuffleChk->blockSignals(false);
        m_splitShuffleChk->blockSignals(true);
        m_splitShuffleChk->setChecked(m_splitShuffle);
        m_splitShuffleChk->blockSignals(false);
        m_trainPctSpin->blockSignals(true);
        m_trainPctSpin->setValue(m_trainPct);
        m_trainPctSpin->blockSignals(false);
        m_valPctSpin->blockSignals(true);
        m_valPctSpin->setValue(m_valPct);
        m_valPctSpin->blockSignals(false);
        m_normCombo->blockSignals(true);
        m_normCombo->setCurrentIndex(m_normMode);
        m_normCombo->blockSignals(false);
        m_muRow->setVisible(m_opt == OPT_MOMENTUM);
        m_adamRow->setVisible(m_opt == OPT_ADAM);
    }

    void applyPresetXor(bool refresh) {
        m_dataset = DS_XOR;
        m_numHidden = 1;
        m_hiddenN[0] = 8;
        m_hiddenAct[0] = ACT_TANH;
        m_outputAct = "sigmoid";
        m_loss = LOSS_MSE;
        m_opt = OPT_ADAM;
        m_lr = 0.05;
        m_epochsTarget = 1500;
        m_batchIdx = 0;
        m_seed = 42;
        m_shuffle = true;
        m_normMode = 1;
        pushUiFromState();
        if (refresh) configChanged();
    }
    void applyPresetIris() {
        m_dataset = DS_IRIS;
        m_numHidden = 2;
        m_hiddenN[0] = 8; m_hiddenN[1] = 8;
        m_hiddenAct[0] = ACT_RELU; m_hiddenAct[1] = ACT_RELU;
        m_outputAct = "sigmoid";
        m_loss = LOSS_CCE;
        m_opt = OPT_ADAM;
        m_lr = 0.01;
        m_epochsTarget = 2000;
        m_batchIdx = 2; // 16
        m_seed = 42;
        m_shuffle = true;
        m_normMode = 1;
        pushUiFromState();
        configChanged();
    }
    void applyPresetBinary() {
        m_dataset = DS_AND;
        m_numHidden = 1;
        m_hiddenN[0] = 4;
        m_hiddenAct[0] = ACT_TANH;
        m_outputAct = "sigmoid";
        m_loss = LOSS_BCE;
        m_opt = OPT_SGD;
        m_lr = 0.01;
        m_epochsTarget = 1000;
        m_batchIdx = 0;
        m_seed = 42;
        m_shuffle = true;
        m_normMode = 1;
        pushUiFromState();
        configChanged();
    }

    void applyOptimizerDefaults() {
        if (m_opt == OPT_SGD) m_lr = 0.01;
        else if (m_opt == OPT_MOMENTUM) m_lr = 0.01;
        else m_lr = 0.05;
        m_lrSpin->blockSignals(true);
        m_lrSpin->setValue(m_lr);
        m_lrSpin->blockSignals(false);
        m_muRow->setVisible(m_opt == OPT_MOMENTUM);
        m_adamRow->setVisible(m_opt == OPT_ADAM);
    }
};

int main(int argc, char* argv[]) {
    QApplication::setStyle(QStringLiteral("Fusion"));
    QApplication app(argc, argv);
    MainWindow w;
    // Fit smaller screens: clamp the 1420x910 design into the available area.
    QRect avail = QGuiApplication::primaryScreen()->availableGeometry();
    w.resize(std::min(1450, avail.width() - 40), std::min(940, avail.height() - 60));
    w.show();
    // Test hook (no moc needed): --shot <png> --dump <txt> renders the live
    // window offscreen state to disk after startup settles, then quits.
    QString shotPath, dumpPath;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--shot" && i + 1 < argc) shotPath = QString::fromLocal8Bit(argv[++i]);
        if (std::string(argv[i]) == "--dump" && i + 1 < argc) dumpPath = QString::fromLocal8Bit(argv[++i]);
    }
    if (!shotPath.isEmpty() || !dumpPath.isEmpty()) {
        QTimer::singleShot(2000, [&]() {
            if (!dumpPath.isEmpty()) w.dumpDebug(dumpPath);
            if (!shotPath.isEmpty()) w.grab().save(shotPath);
            app.quit();
        });
    }
    int rc = app.exec();
    if (g_trainThread && g_trainThread->joinable()) g_trainThread->join();
    return rc;
}
