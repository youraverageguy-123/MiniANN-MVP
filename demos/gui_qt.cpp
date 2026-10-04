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
#include <QPointer>
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
#include <QScrollBar>
#include <QCloseEvent>
#include <QMimeData>
#include <QUrl>
#include <QGuiApplication>
#include <QScreen>
#include <QStyleFactory>
#include <QScrollArea>
#include <QFile>
#include <QDir>
#include <QTextStream>
#include <QFileInfo>
#include <QDesktopServices>

#include "miniann/experiment.hpp"
#include "miniann/serializer.hpp"
#include "miniann/report.hpp"
#include "miniann/visualizer.hpp"

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
    std::atomic<int> epochDelayMs{0};
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
    bool lastHasVal = false, lastTiny = false;
    std::string lastWarning, lastArch, lastOptDesc, lastDsDesc, lastLoss;
    std::string lastBatchTxt;
    int lastSeed = 42;
    std::vector<std::vector<std::size_t>> lastConfusion;
    std::size_t lastNumClasses = 0;
    bool lastHasConfusion = false;
    std::shared_ptr<NeuralNetwork> lastNet;
    Dataset lastTrain, lastTest;
    std::size_t lastInDim = 0, lastOutDim = 0;

    void onEpoch(int epoch, const TrainingHistory& hist) override {
        int delay = epochDelayMs.load();
        if (delay > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        }
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

    void copyFrom(const PlotWidget& o) {
        m_train = o.m_train;
        m_val = o.m_val;
        m_accuracy = o.m_accuracy;
        m_hasVal = o.m_hasVal;
        m_liveEpoch = o.m_liveEpoch;
        update();
    }

    void mouseDoubleClickEvent(QMouseEvent*) override {
        if (m_train.size() < 2) return;
        if (m_expandedDlg) {
            m_expandedDlg->raise();
            m_expandedDlg->activateWindow();
            return;
        }
        auto* dlg = new QDialog(window());
        m_expandedDlg = dlg;
        dlg->setWindowTitle(m_accuracy ? QStringLiteral("Accuracy curves") : QStringLiteral("Loss curves"));
        dlg->setAttribute(Qt::WA_DeleteOnClose);
        dlg->resize(800, 560);
        auto* lay = new QVBoxLayout(dlg);
        lay->setContentsMargins(6, 6, 6, 6);
        auto* big = new PlotWidget;
        big->copyFrom(*this);
        lay->addWidget(big);
        dlg->show();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing, true);
        p.fillRect(rect(), QColor(0x05, 0x07, 0x0C));

        const int ml = 52, mr = 14, mt = 32, mb = 28;
        QRect area(ml, mt, width() - ml - mr, height() - mt - mb);
        if (area.width() < 40 || area.height() < 40) return;

        p.setPen(QPen(QColor(0x1E, 0x29, 0x3B), 1));
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
    QPointer<QDialog> m_expandedDlg;
};

// ------------------------------------------------------- decision boundary
class BoundaryWidget : public QWidget {
public:
    explicit BoundaryWidget(QWidget* parent = nullptr) : QWidget(parent) {
        setMinimumHeight(200);
        setMouseTracking(true);
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

    // Live predictor for hover tooltips (trained net, may be null).
    void setPredictor(std::shared_ptr<NeuralNetwork> net, double thresh = 0.5) {
        m_net = std::move(net);
        m_thresh = thresh;
    }

    void copyFrom(const BoundaryWidget& o) {
        m_grid = o.m_grid;
        m_trainPts = o.m_trainPts;
        m_trainCls = o.m_trainCls;
        m_testPts = o.m_testPts;
        m_testCls = o.m_testCls;
        m_x0 = o.m_x0; m_x1 = o.m_x1; m_y0 = o.m_y0; m_y1 = o.m_y1;
        m_ready = o.m_ready;
        m_net = o.m_net;
        m_thresh = o.m_thresh;
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing, true);
        p.fillRect(rect(), QColor(0x05, 0x07, 0x0C));
        const int m = 14;
        QRect area(m, m + 16, width() - 2 * m, height() - 2 * m - 22);
        if (area.width() < 40 || area.height() < 40) return;
        p.setPen(QPen(QColor(0x1E, 0x29, 0x3B), 1));
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
        if (m_hover.x() >= 0 && m_ready && !m_grid.isNull()) {
            p.setPen(QPen(QColor(255, 255, 255, 90), 1));
            p.drawLine(m_hover.x(), area.top(), m_hover.x(), area.bottom());
            p.drawLine(area.left(), m_hover.y(), area.right(), m_hover.y());
        }
    }

    QRect plotArea() const {
        const int m = 14;
        return QRect(m, m + 16, width() - 2 * m, height() - 2 * m - 22);
    }

    void mouseMoveEvent(QMouseEvent* e) override {
        m_hover = QPoint(-1, -1);
        QRect area = plotArea();
        if (!m_ready || m_grid.isNull() || !area.contains(e->pos())) {
            QToolTip::hideText();
            update();
            return;
        }
        double dx = e->pos().x() - area.left();
        double dy = area.bottom() - e->pos().y();
        double x1 = m_x0 + dx / std::max(1, area.width()) * (m_x1 - m_x0);
        double x2 = m_y0 + dy / std::max(1, area.height()) * (m_y1 - m_y0);
        QString tip = QString::asprintf("x1=%.3f  x2=%.3f", x1, x2);
        if (m_net && m_net->numLayers() > 0) {
            Vector out = m_net->predict({x1, x2});
            if (out.size() == 1) {
                int c = out[0] >= m_thresh ? 1 : 0;
                double conf = c ? out[0] : 1.0 - out[0];
                tip += QString::asprintf("\n→ class %d  (%.1f%%)", c, conf * 100.0);
            } else if (!out.empty()) {
                std::size_t b = 0;
                for (std::size_t k = 1; k < out.size(); ++k)
                    if (out[k] > out[b]) b = k;
                tip += QString::asprintf("\n→ class %llu  (%.1f%%)",
                    (unsigned long long)b, out[b] * 100.0);
            }
        }
        // Nearest training point, so hovering doubles as a data inspector.
        double best = 1e300;
        int bestC = -1;
        bool bestTest = false;
        auto consider = [&](const std::vector<QPointF>& pts, const std::vector<int>& cls, bool test) {
            for (std::size_t i = 0; i < pts.size(); ++i) {
                double ddx = pts[i].x() - x1, ddy = pts[i].y() - x2;
                double d = ddx * ddx + ddy * ddy;
                if (d < best) {
                    best = d;
                    bestC = (i < cls.size()) ? cls[i] : -1;
                    bestTest = test;
                }
            }
        };
        consider(m_trainPts, m_trainCls, false);
        consider(m_testPts, m_testCls, true);
        if (bestC >= 0)
            tip += QString::asprintf("\nnearest: class %d (%.3f away, %s)", bestC,
                std::sqrt(best), bestTest ? "test" : "train");
        m_hover = e->pos();
        QToolTip::showText(e->globalPosition().toPoint(), tip, this);
        update();
    }

    void leaveEvent(QEvent*) override {
        m_hover = QPoint(-1, -1);
        QToolTip::hideText();
        update();
    }

    void mouseDoubleClickEvent(QMouseEvent*) override {
        if (!m_ready || m_grid.isNull()) return;
        if (m_expandedDlg) {
            m_expandedDlg->raise();
            m_expandedDlg->activateWindow();
            return;
        }
        auto* dlg = new QDialog(window());
        m_expandedDlg = dlg;
        dlg->setWindowTitle(QStringLiteral("Decision boundary"));
        dlg->setAttribute(Qt::WA_DeleteOnClose);
        dlg->resize(800, 620);
        auto* lay = new QVBoxLayout(dlg);
        lay->setContentsMargins(6, 6, 6, 6);
        auto* big = new BoundaryWidget;
        big->copyFrom(*this);
        lay->addWidget(big);
        dlg->show();
    }

private:
    QImage m_grid;
    std::vector<QPointF> m_trainPts, m_testPts;
    std::vector<int> m_trainCls, m_testCls;
    double m_x0 = 0, m_x1 = 1, m_y0 = 0, m_y1 = 1;
    bool m_ready = false;
    std::shared_ptr<NeuralNetwork> m_net;
    double m_thresh = 0.5;
    QPoint m_hover{-1, -1};
    QPointer<QDialog> m_expandedDlg;
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
        setMouseTracking(true);
    }
    // Skeleton preview from config (no weights yet).
    void setArch(ArchDesc a) {
        m_arch = std::move(a);
        m_net.reset();
        m_tag = QStringLiteral("PREVIEW — edit config, then START TRAINING");
        update();
    }
    // Trained net: nodes show live firing on the probe input.
    void setNet(std::shared_ptr<NeuralNetwork> net, Vector probe, QString tag) {
        m_net = std::move(net);
        m_probe = std::move(probe);
        m_tag = std::move(tag);
        m_arch.valid = false;
        update();
    }
    void copyFrom(const NetWidget& o) {
        m_arch = o.m_arch;
        m_net = o.m_net;
        m_probe = o.m_probe;
        m_tag = o.m_tag;
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing, true);
        p.fillRect(rect(), QColor(0x05, 0x07, 0x0C));
        const int m = 14;

        // Top tag banner (strictly outside and above the neuron area)
        if (!m_tag.isEmpty()) {
            QRect tagBox(m, m, width() - 2 * m, 24);
            p.setPen(QPen(QColor(0x1E, 0x29, 0x3B), 1));
            p.setBrush(QColor(0x0A, 0x0E, 0x17));
            p.drawRect(tagBox);

            QFont tf = font();
            tf.setPointSize(8);
            tf.setBold(true);
            p.setFont(tf);
            p.setPen(QColor(0x94, 0xA3, 0xB8));
            QString tag = fontMetrics().elidedText(m_tag, Qt::ElideRight, tagBox.width() - 16);
            p.drawText(tagBox.adjusted(10, 0, -10, 0), Qt::AlignLeft | Qt::AlignVCenter, tag);
        }

        // Bounding area for network nodes and edges: starts strictly below the tag banner
        QRect area(m, m + 30, width() - 2 * m, height() - 2 * m - 30 - 26);
        if (area.width() < 40 || area.height() < 40) return;
        p.setPen(QPen(QColor(0x1E, 0x29, 0x3B), 1));
        p.setBrush(Qt::NoBrush);
        p.drawRect(area.adjusted(0, 0, -1, -1));

        // Resolve layers to draw: trained net wins, else skeleton.
        struct Col { QString name; std::size_t n; };
        std::vector<Col> cols;
        if (m_net && m_net->numLayers() > 0) {
            cols.push_back({QStringLiteral("Input"), m_net->layers()[0].inputSize()});
            for (std::size_t l = 0; l < m_net->numLayers(); ++l) {
                const auto& layer = m_net->layers()[l];
                QString an = layer.neurons().empty()
                    ? QStringLiteral("?")
                    : QString::fromStdString(layer.neurons()[0].activation().name());
                bool last = (l + 1 == m_net->numLayers());
                cols.push_back({(last ? QStringLiteral("Output·") : QStringLiteral("H%1·").arg(l + 1)) + an,
                                last ? layer.size() : layer.size()});
            }
        } else if (m_arch.valid && m_arch.sizes.size() >= 2) {
            for (std::size_t l = 0; l < m_arch.sizes.size(); ++l)
                cols.push_back({m_arch.names[l], m_arch.sizes[l]});
        } else {
            p.setPen(QColor(0x8A, 0x90, 0xA0));
            p.drawText(area, Qt::AlignCenter,
                       QStringLiteral("Train a network (or load a model)\nto see its architecture diagram."));
            return;
        }

        // Firing values for the probe (trained net only).
        std::vector<std::vector<double>> fire;
        if (m_net && !m_probe.empty()) {
            Vector cur = m_probe;
            fire.push_back(cur);
            for (auto& layer : m_net->layers()) {
                cur = layer.forward(cur);
                fire.push_back(cur);
            }
        }
        const std::size_t L = cols.size();
        const std::size_t kShowMax = 8; // declutter: never draw more than 8 nodes/column
        // Column centers are inset half a slot so first/last labels stay inside.
        const double slot = (double)area.width() / (double)L;
        auto colX = [&](std::size_t l) {
            return (L == 1) ? area.center().x() : area.left() + slot * ((double)l + 0.5);
        };
        std::vector<std::vector<QPointF>> pts(L);
        std::vector<std::size_t> shown(L);
        m_hit.assign(L, {});
        for (std::size_t l = 0; l < L; ++l) {
            shown[l] = std::min<std::size_t>(cols[l].n, kShowMax);
            double cx = colX(l);
            for (std::size_t i = 0; i < shown[l]; ++i) {
                double cy = (shown[l] == 1) ? area.center().y()
                    : area.top() + 20 + (double)i / (double)(shown[l] - 1) * (area.height() - 40);
                pts[l].push_back(QPointF(cx, cy));
                m_hit[l].push_back(QRectF(cx - 12, cy - 12, 24, 24));
            }
        }

        // Edges: pure black-bluish visual (luminous cyan/blue lines scaled by weight magnitude)
        for (std::size_t l = 1; l < L; ++l) {
            for (std::size_t a_idx = 0; a_idx < shown[l - 1]; ++a_idx) {
                for (std::size_t b_idx = 0; b_idx < shown[l]; ++b_idx) {
                    double w = 0.0;
                    bool hasW = false;
                    if (m_net && (l - 1) < m_net->numLayers()) {
                        const auto& lyr = m_net->layers()[l - 1];
                        if (b_idx < lyr.size()) {
                            const auto& ws = lyr.neurons()[b_idx].weights();
                            if (a_idx < ws.size()) {
                                w = ws[a_idx];
                                hasW = true;
                            }
                        }
                    }

                    if (hasW) {
                        double absW = std::abs(w);
                        qreal penW = std::clamp(1.0 + absW * 0.4, 1.0, 2.4);
                        int alpha = std::clamp(static_cast<int>(30 + absW * 55), 30, 180);
                        p.setPen(QPen(QColor(0x5A, 0xA9, 0xE6, alpha), penW));
                    } else {
                        p.setPen(QPen(QColor(0x5A, 0xA9, 0xE6, 30), 1));
                    }
                    p.drawLine(pts[l - 1][a_idx], pts[l][b_idx]);
                }
            }
        }

        // Nodes: classic luminous blue firing on deep navy/black base
        for (std::size_t l = 0; l < L; ++l) {
            for (std::size_t i = 0; i < shown[l]; ++i) {
                double f = -1.0;
                if (l < fire.size() && i < fire[l].size()) {
                    f = fire[l][i];
                    if (fire[l].size() > 1) f = std::max(0.0, std::min(1.0, f));
                }
                QColor fill = (f < 0.0) ? QColor(0x1B, 0x1E, 0x2B)
                    : QColor::fromHslF(0.58, 0.75, 0.12 + 0.35 * std::max(0.0, std::min(1.0, f)));
                p.setBrush(fill);
                p.setPen(QPen(QColor(0x7F, 0xB3, 0xE8), 1.5));
                p.drawEllipse(pts[l][i], 9, 9);
            }
            if (shown[l] < cols[l].n) {
                p.setPen(QColor(0x8A, 0x90, 0xA0));
                QFont f = font();
                f.setPointSize(8);
                p.setFont(f);
                double cx = pts[l][0].x();
                p.drawText(QRect(int(cx) - 40, area.bottom() - 30, 80, 16), Qt::AlignHCenter,
                           QStringLiteral("+") + QString::number((unsigned long long)(cols[l].n - shown[l])));
            }
        }

        // Column labels: short, elided to the column slot (no overlap).
        QFont lf = font();
        lf.setPointSize(8);
        p.setFont(lf);
        p.setPen(QColor(0xD0, 0xD3, 0xDB));
        for (std::size_t l = 0; l < L; ++l) {
            double cx = colX(l);
            QString t = cols[l].name + QString::asprintf(" [%llu]", (unsigned long long)cols[l].n);
            t = fontMetrics().elidedText(t, Qt::ElideRight, int(slot) - 4);
            int w = fontMetrics().horizontalAdvance(t);
            p.drawText(QRect(int(cx) - w / 2, area.bottom() + 4, w + 4, 18), Qt::AlignLeft, t);
        }
    }

    void mouseMoveEvent(QMouseEvent* e) override {
        if (!m_net || m_hit.empty()) {
            QToolTip::hideText();
            return;
        }
        for (std::size_t l = 0; l < m_hit.size(); ++l)
            for (std::size_t i = 0; i < m_hit[l].size(); ++i)
                if (m_hit[l][i].contains(e->pos())) {
                    QString tip;
                    if (l == 0) {
                        double v = (i < m_probe.size()) ? m_probe[i] : 0.0;
                        tip = QString::asprintf("Input Node #%llu\nValue: %.4f", (unsigned long long)i + 1, v);
                    } else {
                        std::size_t li = l - 1;
                        if (li < m_net->numLayers() && i < m_net->layers()[li].size()) {
                            const Neuron& n = m_net->layers()[li].neurons()[i];
                            QString ws;
                            for (std::size_t k = 0; k < std::min<std::size_t>(n.weights().size(), 6); ++k)
                                ws += QString::asprintf("%s%.3f", k ? ", " : "", n.weights()[k]);
                            if (n.weights().size() > 6) ws += QStringLiteral(", …");
                            double out = n.lastOutput();
                            tip = QString::asprintf("Layer %llu, Neuron %llu\nBias: %+.4f\nAct: %s | Out: %.4f\nWeights: [%s]",
                                (unsigned long long)li + 1, (unsigned long long)i + 1,
                                n.bias(), n.activation().name().c_str(), out, ws.toLatin1().constData());
                        }
                    }
                    if (!tip.isEmpty()) QToolTip::showText(e->globalPosition().toPoint(), tip, this);
                    return;
                }
        QToolTip::hideText();
    }

    void leaveEvent(QEvent*) override { QToolTip::hideText(); }

    void mouseDoubleClickEvent(QMouseEvent*) override {
        if (m_expandedDlg) {
            m_expandedDlg->raise();
            m_expandedDlg->activateWindow();
            return;
        }
        auto* dlg = new QDialog(window());
        m_expandedDlg = dlg;
        dlg->setWindowTitle(m_tag.isEmpty() ? QStringLiteral("Network Architecture") : m_tag);
        dlg->setAttribute(Qt::WA_DeleteOnClose);
        dlg->resize(900, 650);
        auto* lay = new QVBoxLayout(dlg);
        lay->setContentsMargins(6, 6, 6, 6);
        auto* big = new NetWidget;
        big->copyFrom(*this);
        lay->addWidget(big);
        dlg->show();
    }

private:
    ArchDesc m_arch;
    std::shared_ptr<NeuralNetwork> m_net;
    Vector m_probe;
    QString m_tag;
    std::vector<std::vector<QRectF>> m_hit;
    QPointer<QDialog> m_expandedDlg;
};

// ---------------------------------------------------------------- main window
// Collapsible left-panel group: one-line header, content hidden on demand.
// Minimal look — a collapsed group is a single row, an expanded one is flat.
class CollapsibleSection : public QWidget {
public:
    CollapsibleSection(const QString& title, bool expanded, QWidget* parent = nullptr)
        : QWidget(parent), m_title(title) {
        auto* outer = new QVBoxLayout(this);
        outer->setContentsMargins(0, 4, 0, 4);
        outer->setSpacing(6);

        auto* line = new QFrame;
        line->setFixedHeight(1);
        line->setStyleSheet(QStringLiteral("background:#1E293B; border:none;"));
        outer->addWidget(line);

        m_head = new QPushButton;
        m_head->setCursor(Qt::PointingHandCursor);
        m_head->setFlat(true);
        m_head->setMinimumHeight(26);
        m_head->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        m_head->setStyleSheet(QStringLiteral(
            "QPushButton { text-align:left; background:transparent; border:none;"
            " border-radius:0px; padding:4px 6px; color:#94A3B8; font-weight:700; font-size:11px; letter-spacing:0.5px; }"
            "QPushButton:hover { background:rgba(255,255,255,0.04); color:#F1F5F9; }"));
        connect(m_head, &QPushButton::clicked, [this]() { setExpanded(!m_expanded); });
        outer->addWidget(m_head);

        m_body = new QWidget;
        m_body->setStyleSheet(QStringLiteral("background:transparent; border:none;"));
        m_lay = new QVBoxLayout(m_body);
        m_lay->setContentsMargins(0, 0, 0, 0);
        m_lay->setSpacing(8);
        outer->addWidget(m_body);
        setExpanded(expanded);
    }
    QVBoxLayout* content() { return m_lay; }
    void setExpanded(bool e) {
        m_expanded = e;
        m_body->setVisible(e);
        QString disp = m_title;
        disp.replace(QStringLiteral("&"), QStringLiteral("&&"));
        m_head->setText((m_expanded ? QString::fromUtf8("▾  ") : QString::fromUtf8("▸  ")) + disp);
    }

private:
    QString m_title;
    QPushButton* m_head = nullptr;
    QWidget* m_body = nullptr;
    QVBoxLayout* m_lay = nullptr;
    bool m_expanded = true;
};

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
        if (QApplication::arguments().contains(QStringLiteral("--net"))) {
            m_view = VIEW_NETWORK;
            syncTabs();
        }
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
        if (m_hiddenCfgBtn) geo("hiddenCfg", m_hiddenCfgBtn);
        QString dir = QFileInfo(path).absolutePath() + QStringLiteral("/");
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
    int m_normMode = 1;
    struct HiddenCfg { int n = 8; ActiveActivation act = ACT_RELU; };
    static constexpr int kMaxHidden = 8;
    std::vector<HiddenCfg> m_hidden = {{8, ACT_TANH}, {8, ACT_RELU}};
    UiState m_state = ST_IDLE;
    bool m_resultsShownFor = false;
    unsigned m_seenSeq = 0;

    // ---- dataset preview (recomputed when the data fingerprint changes) ----
    std::string m_previewFp;
    PreparedData m_preview;
    std::string m_previewErr;
    // Fingerprint of the config whose net is currently drawn (trained or
    // preview). A stale trained net never poses as the current config.
    std::string m_shownNetFp;

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
    QLabel* m_hiddenSumLbl = nullptr;
    QPushButton* m_hiddenCfgBtn = nullptr;
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
    QComboBox* m_normCombo = nullptr;
    std::vector<QPushButton*> m_lossBtns;
    std::vector<QPushButton*> m_optBtns;
    QLabel* m_warnLbl = nullptr;
    QLabel* m_cfgLbl = nullptr;
    QLabel* m_summaryLbl = nullptr;
    QLabel* m_progressLbl = nullptr;
    QPushButton* m_trainBtn = nullptr;
    QPushButton* m_saveBtn = nullptr, *m_loadBtn = nullptr, *m_reportBtn = nullptr;
    QPushButton* m_tabLoss = nullptr, *m_tabAcc = nullptr, *m_tabBnd = nullptr, *m_tabNet = nullptr;
    PlotWidget* m_plot = nullptr;
    BoundaryWidget* m_boundary = nullptr;
    NetWidget* m_netview = nullptr;
    QWidget* m_plotStack = nullptr;
    QLabel* m_summaryBar = nullptr;
    QTimer* m_timer = nullptr;
    QSlider* m_speedSlider = nullptr;
    QPushButton* m_stepBtn = nullptr;
    QScrollArea* m_cfgScroll = nullptr;

    static void checkOnly(const std::vector<QPushButton*>& v, int idx) {
        for (int i = 0; i < (int)v.size(); ++i)
            v[(std::size_t)i]->setChecked(i == idx);
    }

    static void setExclusive(const std::vector<QPushButton*>& v) {
        for (QPushButton* b : v) {
            b->setCheckable(true);
            QObject::connect(b, &QPushButton::clicked, [v, b]() {
                for (QPushButton* o : v) o->setChecked(o == b);
            });
        }
        if (!v.empty()) v.front()->setChecked(true);
    }

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
        for (int i = 0; i < (int)m_hidden.size(); ++i) {
            c.hidden.push_back(m_hidden[(std::size_t)i].n);
            c.hiddenActs.push_back(actToName(m_hidden[(std::size_t)i].act));
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
               std::to_string(m_targetCol->value()) + "|" + (m_headerChk->isChecked() ? "h" : "n");
    }

    void refreshPreview() {
        std::string fp = fingerprint();
        if (fp == m_previewFp) return;
        m_previewFp = fp;
        m_previewErr.clear();
        try {
            if (m_dataset == DS_IRIS && !QFile::exists(QStringLiteral("data/iris_small.csv"))) {
                m_previewErr = "data/iris_small.csv not found (working dir is " +
                    QDir::currentPath().toStdString() + ") — launch gui_qt.exe from MiniANN_MVP.";
                m_preview = PreparedData();
                qWarning("refreshPreview: %s", m_previewErr.c_str());
                return;
            }
            m_preview = ExperimentController::prepare(currentConfig());
        } catch (const std::exception& e) {
            m_preview = PreparedData();
            m_previewErr = e.what();
            qWarning("refreshPreview: %s", m_previewErr.c_str());
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
        for (std::size_t i = 0; i < m_hidden.size(); ++i)
            s += " -> " + std::to_string(m_hidden[i].n) + "(" + shortAct(actToName(m_hidden[i].act)) + ")";
        s += " -> " + outT + "(" + shortAct(m_outputAct) + ")";
        return s;
    }

    void buildUi() {
        // ----- top bar: every metric is labeled with its source (§2.1) -----
        auto* top = new QFrame;
        top->setObjectName(QStringLiteral("topbar"));
        auto* topLay = new QHBoxLayout(top);
        topLay->setContentsMargins(20, 10, 20, 10);
        topLay->setSpacing(10);
        auto* titleLbl = new QLabel(QStringLiteral("MiniANN MVP - Network Workbench"));
        QFont titleFont = titleLbl->font();
        titleFont.setPointSize(14);
        titleFont.setBold(true);
        titleLbl->setFont(titleFont);
        titleLbl->setStyleSheet(QStringLiteral("color:#F8FAFC;"));
        topLay->addWidget(titleLbl);
        topLay->addStretch(1);
        m_epochLbl = new QLabel;
        m_trainLossLbl = new QLabel;
        m_trainAccLbl = new QLabel;
        m_valAccLbl = new QLabel;
        m_testAccLbl = new QLabel;
        auto makeStatChip = [](QLabel* lbl, const char* fg) {
            lbl->setStyleSheet(QString::asprintf(
                "QLabel { background:#0A0E17; color:%s; border:1px solid #1E293B; border-radius:0px; padding:5px 11px; font-family:'Consolas','Segoe UI',monospace; font-size:11px; font-weight:bold; }",
                fg));
        };
        makeStatChip(m_epochLbl, "#F1F5F9");
        makeStatChip(m_trainLossLbl, "#FBBF24");
        makeStatChip(m_trainAccLbl, "#38BDF8");
        makeStatChip(m_valAccLbl, "#F59E0B");
        makeStatChip(m_testAccLbl, "#34D399");
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
        left->setFixedWidth(480);
        auto* outer = new QVBoxLayout(left);
        outer->setContentsMargins(20, 18, 20, 18);
        outer->setSpacing(10);
        m_configBox = new QWidget;
        m_configBox->setStyleSheet(QStringLiteral("background:transparent; border:none;"));
        auto* lv = new QVBoxLayout(m_configBox);
        lv->setContentsMargins(0, 0, 0, 0);
        lv->setSpacing(10);
        // The config column (~1150px of controls) is taller than short
        // windows: without scrolling, QVBoxLayout squeezes word-wrap labels
        // to 0px and overlaps rows, which painted as dotted/ghosted text.
        m_cfgScroll = new QScrollArea;
        m_cfgScroll->setWidgetResizable(true);
        m_cfgScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        m_cfgScroll->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        m_cfgScroll->setFrameShape(QFrame::NoFrame);
        m_cfgScroll->setWidget(m_configBox);
        outer->addWidget(m_cfgScroll, 1);

        // presets
        auto* preRow = new QHBoxLayout;
        preRow->setSpacing(8);
        auto* preLbl = new QLabel(QStringLiteral("PRESETS"));
        preLbl->setStyleSheet(QStringLiteral("color:#64748B; font-weight:700; font-size:10px; letter-spacing:0.5px;"));
        preRow->addWidget(preLbl);
        auto* preWell = new QFrame;
        preWell->setStyleSheet(QStringLiteral("background:#080C14; border:1px solid #1E293B; border-radius:0px;"));
        auto* preWellLay = new QHBoxLayout(preWell);
        preWellLay->setContentsMargins(2, 2, 2, 2);
        preWellLay->setSpacing(2);
        const char* preNames[3] = {"XOR", "Iris", "Binary"};
        for (int i = 0; i < 3; ++i) {
            QPushButton* b = new QPushButton(QString::fromLatin1(preNames[i]));
            b->setCursor(Qt::PointingHandCursor);
            b->setFixedHeight(26);
            b->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
            b->setStyleSheet(QStringLiteral(
                "QPushButton { background:transparent; color:#94A3B8; border:none; border-radius:0px; font-size:11px; font-weight:600; padding:4px 8px; }"
                "QPushButton:hover { background:rgba(255,255,255,0.06); color:#F1F5F9; }"
                "QPushButton:pressed { background:#0284C7; color:#FFFFFF; }"));
            preWellLay->addWidget(b);
            m_presetBtns.push_back(b);
        }
        connect(m_presetBtns[0], &QPushButton::clicked, [this]() { applyPresetXor(true); });
        connect(m_presetBtns[1], &QPushButton::clicked, [this]() { applyPresetIris(); });
        connect(m_presetBtns[2], &QPushButton::clicked, [this]() { applyPresetBinary(); });
        preRow->addWidget(preWell, 1);
        lv->addLayout(preRow);

        // 1. dataset
        auto* sec1 = new CollapsibleSection(QStringLiteral("1. DATASET"), true, m_configBox);
        lv->addWidget(sec1);
        QVBoxLayout* s1 = sec1->content();
        auto* dsWell = new QFrame;
        dsWell->setStyleSheet(QStringLiteral("background:#080C14; border:1px solid #1E293B; border-radius:0px;"));
        auto* dsWellLay = new QHBoxLayout(dsWell);
        dsWellLay->setContentsMargins(2, 2, 2, 2);
        dsWellLay->setSpacing(2);
        for (int i = 0; i < 5; ++i) {
            QPushButton* b = new QPushButton(QString::fromLatin1(kDsNames[i]));
            b->setCheckable(true);
            b->setCursor(Qt::PointingHandCursor);
            b->setFixedHeight(28);
            b->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
            b->setStyleSheet(QStringLiteral(
                "QPushButton { background:transparent; color:#94A3B8; border:none; border-radius:0px; font-size:11px; font-weight:600; padding:4px 6px; }"
                "QPushButton:hover { background:rgba(255,255,255,0.06); color:#F1F5F9; }"
                "QPushButton:checked { background:#0284C7; color:#FFFFFF; border:1px solid #38BDF8; font-weight:bold; }"));
            dsWellLay->addWidget(b);
            m_dsBtns.push_back(b);
        }
        setExclusive(m_dsBtns);
        for (int i = 0; i < 5; ++i) {
            connect(m_dsBtns[(std::size_t)i], &QPushButton::clicked, [this, i]() {
                m_dataset = ActiveDataset(i);
                configChanged();
            });
        }
        s1->addWidget(dsWell);

        // dataset info
        auto* dsInfoPanel = new QFrame;
        dsInfoPanel->setObjectName(QStringLiteral("panel"));
        dsInfoPanel->setStyleSheet(QStringLiteral(
            "background:#0A0E17; border:1px solid #1E293B; border-radius:0px;"));
        auto* dsInfoLay = new QVBoxLayout(dsInfoPanel);
        dsInfoLay->setContentsMargins(12, 10, 12, 10);
        dsInfoLay->setSpacing(0);
        m_dsInfoLbl = new QLabel;
        m_dsInfoLbl->setStyleSheet(QStringLiteral("color:#94A3B8; font-size:11px; line-height:140%;"));
        m_dsInfoLbl->setWordWrap(true);
        dsInfoLay->addWidget(m_dsInfoLbl);
        s1->addWidget(dsInfoPanel);

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
        s1->addWidget(m_csvRow);

        // 2. architecture (§5)
        auto* sec2 = new CollapsibleSection(QStringLiteral("2. ARCHITECTURE"), true, m_configBox);
        lv->addWidget(sec2);
        QVBoxLayout* s2 = sec2->content();

        auto* archCard = new QFrame;
        archCard->setStyleSheet(QStringLiteral("background:#0D111A; border:1px solid #1E293B; border-radius:0px;"));
        auto* archCardLay = new QVBoxLayout(archCard);
        archCardLay->setContentsMargins(12, 12, 12, 12);
        archCardLay->setSpacing(10);

        // Top row: Layers count with [-] and [+] steppers
        auto* layRow = new QHBoxLayout;
        layRow->setSpacing(8);
        m_layersLbl = new QLabel;
        m_layersLbl->setStyleSheet(QStringLiteral("color:#F1F5F9; font-weight:600; font-size:12px;"));
        layRow->addWidget(m_layersLbl);
        layRow->addStretch(1);

        auto* stepWell = new QFrame;
        stepWell->setStyleSheet(QStringLiteral("background:#080C14; border:1px solid #1E293B; border-radius:0px;"));
        auto* stepWellLay = new QHBoxLayout(stepWell);
        stepWellLay->setContentsMargins(2, 2, 2, 2);
        stepWellLay->setSpacing(2);

        auto* layMinus = new QPushButton(QStringLiteral("−"));
        auto* layPlus = new QPushButton(QStringLiteral("+"));
        for (auto* btn : {layMinus, layPlus}) {
            btn->setFixedSize(28, 26);
            btn->setCursor(Qt::PointingHandCursor);
            btn->setStyleSheet(QStringLiteral(
                "QPushButton { background:transparent; color:#94A3B8; border:none; border-radius:0px; font-size:14px; font-weight:bold; }"
                "QPushButton:hover { background:rgba(255,255,255,0.08); color:#FFFFFF; }"
                "QPushButton:pressed { background:#0284C7; color:#FFFFFF; }"));
            stepWellLay->addWidget(btn);
        }
        connect(layMinus, &QPushButton::clicked, [this]() {
            if (m_hidden.size() > 1) {
                m_hidden.pop_back();
                configChanged();
            }
        });
        connect(layPlus, &QPushButton::clicked, [this]() {
            if ((int)m_hidden.size() < kMaxHidden) {
                m_hidden.push_back({8, ACT_RELU});
                configChanged();
            }
        });
        layRow->addWidget(stepWell);
        archCardLay->addLayout(layRow);

        // Architecture summary badge / box
        auto* archSummaryBox = new QFrame;
        archSummaryBox->setStyleSheet(QStringLiteral("background:#06080E; border:1px solid #1E293B; border-radius:0px;"));
        auto* archSumLay = new QVBoxLayout(archSummaryBox);
        archSumLay->setContentsMargins(10, 8, 10, 8);
        archSumLay->setSpacing(4);

        m_archLbl = new QLabel;
        m_archLbl->setStyleSheet(QStringLiteral("color:#38BDF8; font-weight:700; font-family:'Consolas','Segoe UI',monospace; font-size:12px;"));
        m_archLbl->setWordWrap(true);
        archSumLay->addWidget(m_archLbl);

        m_hiddenSumLbl = new QLabel;
        m_hiddenSumLbl->setStyleSheet(QStringLiteral("color:#94A3B8; font-size:11px;"));
        m_hiddenSumLbl->setWordWrap(true);
        archSumLay->addWidget(m_hiddenSumLbl);
        archCardLay->addWidget(archSummaryBox);

        // Topology Preset row
        auto* presetRow = new QHBoxLayout;
        presetRow->setSpacing(8);
        auto* presetLbl = new QLabel(QStringLiteral("Preset:"));
        presetLbl->setStyleSheet(QStringLiteral("color:#94A3B8; font-size:11px; font-weight:600;"));
        presetRow->addWidget(presetLbl);
        presetRow->addStretch(1);
        auto* presetCombo = new QComboBox;
        presetCombo->addItem(QStringLiteral("Custom"));
        presetCombo->addItem(QStringLiteral("Minimal [4]"));
        presetCombo->addItem(QStringLiteral("Standard [8, 8]"));
        presetCombo->addItem(QStringLiteral("Deep [16, 8, 4]"));
        presetCombo->addItem(QStringLiteral("Wide [32]"));
        presetCombo->setCurrentIndex(2); // Standard [8, 8]
        presetCombo->setMinimumWidth(125);
        connect(presetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int idx) {
            if (idx == 1) { // Minimal [4]
                m_hidden = {{4, ACT_RELU}};
            } else if (idx == 2) { // Standard [8, 8]
                m_hidden = {{8, ACT_TANH}, {8, ACT_RELU}};
            } else if (idx == 3) { // Deep [16, 8, 4]
                m_hidden = {{16, ACT_RELU}, {8, ACT_RELU}, {4, ACT_RELU}};
            } else if (idx == 4) { // Wide [32]
                m_hidden = {{32, ACT_RELU}};
            }
            configChanged();
        });
        presetRow->addWidget(presetCombo);
        archCardLay->addLayout(presetRow);

        // Configure button (full width rectangular button)
        m_hiddenCfgBtn = makeBtn(QStringLiteral("Custom Layers Detail…"));
        m_hiddenCfgBtn->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        connect(m_hiddenCfgBtn, &QPushButton::clicked, [this]() { openHiddenDialog(); });
        archCardLay->addWidget(m_hiddenCfgBtn);

        // Output activation row
        auto* outRow = new QHBoxLayout;
        outRow->setSpacing(8);
        auto* outLbl = new QLabel(QStringLiteral("Output Activation"));
        outLbl->setStyleSheet(QStringLiteral("color:#94A3B8; font-size:11px; font-weight:600;"));
        outRow->addWidget(outLbl);
        outRow->addStretch(1);
        m_outActCombo = new QComboBox;
        for (auto n : kOutActLabels) m_outActCombo->addItem(QString::fromLatin1(n));
        m_outActCombo->setCurrentIndex(0);
        m_outActCombo->setMinimumWidth(110);
        connect(m_outActCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int i) {
            m_outputAct = kOutActKeys[std::max(0, std::min(5, i))];
            configChanged();
        });
        outRow->addWidget(m_outActCombo);
        archCardLay->addLayout(outRow);

        s2->addWidget(archCard);

        // 3. training (§9)
        auto* sec3 = new CollapsibleSection(QStringLiteral("3. TRAINING"), true, m_configBox);
        lv->addWidget(sec3);
        QVBoxLayout* s3 = sec3->content();

        auto* trainCard = new QFrame;
        trainCard->setStyleSheet(QStringLiteral("background:#0D111A; border:1px solid #1E293B; border-radius:0px;"));
        auto* trainCardLay = new QVBoxLayout(trainCard);
        trainCardLay->setContentsMargins(12, 12, 12, 12);
        trainCardLay->setSpacing(12);

        // Epochs Group
        auto* epGroup = new QVBoxLayout;
        epGroup->setSpacing(6);
        auto* epTop = new QHBoxLayout;
        epTop->setSpacing(8);
        auto* epLab = new QLabel(QStringLiteral("EPOCHS"));
        epLab->setStyleSheet(QStringLiteral("color:#94A3B8; font-weight:700; font-size:11px; letter-spacing:0.5px;"));
        epTop->addWidget(epLab);
        epTop->addStretch(1);
        m_epochValLbl = new QLabel;
        m_epochValLbl->setStyleSheet(QStringLiteral("color:#38BDF8; background:#06080E; border:1px solid #1E293B; border-radius:0px; padding:2px 8px; font-family:'Consolas',monospace; font-weight:bold; font-size:11px;"));
        epTop->addWidget(m_epochValLbl);
        epGroup->addLayout(epTop);

        auto* epRow = new QHBoxLayout;
        epRow->setSpacing(8);
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
        m_epochSpin->setFixedWidth(80);
        connect(m_epochSpin, &QSpinBox::valueChanged, [this](int v) {
            if (v != m_epochsTarget) {
                m_epochsTarget = v;
                configChanged();
            }
        });
        epRow->addWidget(m_epochSpin);

        auto* epStepWell = new QFrame;
        epStepWell->setStyleSheet(QStringLiteral("background:#080C14; border:1px solid #1E293B; border-radius:0px;"));
        auto* epStepLay = new QHBoxLayout(epStepWell);
        epStepLay->setContentsMargins(2, 2, 2, 2);
        epStepLay->setSpacing(2);

        auto* epMinus = new QPushButton(QStringLiteral("−100"));
        auto* epPlus = new QPushButton(QStringLiteral("+100"));
        for (auto* btn : {epMinus, epPlus}) {
            btn->setFixedHeight(26);
            btn->setCursor(Qt::PointingHandCursor);
            btn->setStyleSheet(QStringLiteral(
                "QPushButton { background:transparent; color:#94A3B8; border:none; border-radius:0px; font-size:11px; font-weight:bold; padding:2px 6px; }"
                "QPushButton:hover { background:rgba(255,255,255,0.08); color:#FFFFFF; }"
                "QPushButton:pressed { background:#0284C7; color:#FFFFFF; }"));
            epStepLay->addWidget(btn);
        }
        connect(epMinus, &QPushButton::clicked, [this]() {
            m_epochsTarget = std::max(1, m_epochsTarget - 100);
            configChanged();
        });
        connect(epPlus, &QPushButton::clicked, [this]() {
            m_epochsTarget = std::min(200000, m_epochsTarget + 100);
            configChanged();
        });
        epRow->addWidget(epStepWell);
        epGroup->addLayout(epRow);
        trainCardLay->addLayout(epGroup);

        // Divider
        auto* div1 = new QFrame;
        div1->setFrameShape(QFrame::HLine);
        div1->setStyleSheet(QStringLiteral("background:#1E2333; max-height:1px; border:none;"));
        trainCardLay->addWidget(div1);

        // Parameters Grid: Batch Size, Normalize, Learning Rate, Seed
        auto* paramGrid = new QGridLayout;
        paramGrid->setHorizontalSpacing(12);
        paramGrid->setVerticalSpacing(10);

        // Batch Size
        auto* bCol = new QVBoxLayout;
        bCol->setSpacing(4);
        auto* bHead = new QHBoxLayout;
        auto* bLbl = new QLabel(QStringLiteral("BATCH SIZE"));
        bLbl->setStyleSheet(QStringLiteral("color:#94A3B8; font-size:10px; font-weight:700; letter-spacing:0.5px;"));
        bHead->addWidget(bLbl);
        bHead->addStretch(1);
        m_effBatchLbl = new QLabel;
        m_effBatchLbl->setStyleSheet(QStringLiteral("color:#64748B; font-size:10px;"));
        bHead->addWidget(m_effBatchLbl);
        bCol->addLayout(bHead);
        m_batchCombo = new QComboBox;
        for (auto n : kBatchLabels) m_batchCombo->addItem(QString::fromLatin1(n));
        m_batchCombo->setCurrentIndex(0);
        connect(m_batchCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int i) {
            m_batchIdx = std::max(0, std::min(6, i));
            configChanged();
        });
        bCol->addWidget(m_batchCombo);
        paramGrid->addLayout(bCol, 0, 0);

        // Normalize
        auto* nCol = new QVBoxLayout;
        nCol->setSpacing(4);
        auto* nLbl = new QLabel(QStringLiteral("NORMALIZE"));
        nLbl->setStyleSheet(QStringLiteral("color:#94A3B8; font-size:10px; font-weight:700; letter-spacing:0.5px;"));
        nCol->addWidget(nLbl);
        m_normCombo = new QComboBox;
        m_normCombo->addItems({QStringLiteral("None"), QStringLiteral("MaxAbs"), QStringLiteral("MinMax")});
        m_normCombo->setCurrentIndex(1);
        connect(m_normCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int i) {
            m_normMode = i;
            configChanged();
        });
        nCol->addWidget(m_normCombo);
        paramGrid->addLayout(nCol, 0, 1);

        // Learning Rate
        auto* lrCol = new QVBoxLayout;
        lrCol->setSpacing(4);
        auto* lrLbl = new QLabel(QStringLiteral("LEARNING RATE"));
        lrLbl->setStyleSheet(QStringLiteral("color:#94A3B8; font-size:10px; font-weight:700; letter-spacing:0.5px;"));
        lrCol->addWidget(lrLbl);
        m_lrSpin = new QDoubleSpinBox;
        m_lrSpin->setRange(0.0001, 1.0);
        m_lrSpin->setDecimals(4);
        m_lrSpin->setSingleStep(0.005);
        m_lrSpin->setValue(0.05);
        connect(m_lrSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [this](double v) {
            m_lr = v;
            configChanged();
        });
        lrCol->addWidget(m_lrSpin);
        paramGrid->addLayout(lrCol, 1, 0);

        // Seed
        auto* sCol = new QVBoxLayout;
        sCol->setSpacing(4);
        auto* sLbl = new QLabel(QStringLiteral("SEED"));
        sLbl->setStyleSheet(QStringLiteral("color:#94A3B8; font-size:10px; font-weight:700; letter-spacing:0.5px;"));
        sCol->addWidget(sLbl);
        m_seedSpin = new QSpinBox;
        m_seedSpin->setRange(0, 999999);
        m_seedSpin->setValue(42);
        connect(m_seedSpin, &QSpinBox::valueChanged, [this](int v) {
            m_seed = v;
            configChanged();
        });
        sCol->addWidget(m_seedSpin);
        paramGrid->addLayout(sCol, 1, 1);

        trainCardLay->addLayout(paramGrid);

        // Shuffle checkbox row
        auto* shufRow = new QHBoxLayout;
        m_shuffleChk = new QCheckBox(QStringLiteral("Shuffle dataset samples every epoch"));
        m_shuffleChk->setChecked(true);
        m_shuffleChk->setStyleSheet(QStringLiteral("color:#C9CDD8; font-size:11px;"));
        connect(m_shuffleChk, &QCheckBox::toggled, [this](bool b) {
            m_shuffle = b;
            configChanged();
        });
        shufRow->addWidget(m_shuffleChk);
        shufRow->addStretch(1);
        trainCardLay->addLayout(shufRow);

        s3->addWidget(trainCard);

        // 4. loss & optimizer (§7, §8) — collapsed by default; warnings stay
        // visible below the section so they are never hidden with it.
        auto* sec4 = new CollapsibleSection(QStringLiteral("4. LOSS & OPTIMIZER"), false, m_configBox);
        lv->addWidget(sec4);
        QVBoxLayout* s4 = sec4->content();

        auto* optCard = new QFrame;
        optCard->setStyleSheet(QStringLiteral("background:#0D111A; border:1px solid #1E293B; border-radius:0px;"));
        auto* optCardLay = new QVBoxLayout(optCard);
        optCardLay->setContentsMargins(12, 12, 12, 12);
        optCardLay->setSpacing(10);

        // Loss selector
        auto* lossLbl = new QLabel(QStringLiteral("LOSS FUNCTION"));
        lossLbl->setStyleSheet(QStringLiteral("color:#94A3B8; font-size:10px; font-weight:700; letter-spacing:0.5px;"));
        optCardLay->addWidget(lossLbl);

        auto* lossWell = new QFrame;
        lossWell->setStyleSheet(QStringLiteral("background:#080C14; border:1px solid #1E293B; border-radius:0px;"));
        auto* lossWellLay = new QHBoxLayout(lossWell);
        lossWellLay->setContentsMargins(2, 2, 2, 2);
        lossWellLay->setSpacing(2);
        for (int i = 0; i < 3; ++i) {
            QPushButton* b = new QPushButton(QString::fromLatin1(kLossLabels[i]));
            b->setCheckable(true);
            b->setCursor(Qt::PointingHandCursor);
            b->setFixedHeight(28);
            b->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
            b->setStyleSheet(QStringLiteral(
                "QPushButton { background:transparent; color:#94A3B8; border:none; border-radius:0px; font-size:11px; font-weight:600; }"
                "QPushButton:hover { background:rgba(255,255,255,0.06); color:#F1F5F9; }"
                "QPushButton:checked { background:#0284C7; color:#FFFFFF; border:1px solid #38BDF8; font-weight:bold; }"));
            lossWellLay->addWidget(b);
            m_lossBtns.push_back(b);
        }
        setExclusive(m_lossBtns);
        for (int i = 0; i < 3; ++i) {
            connect(m_lossBtns[(std::size_t)i], &QPushButton::clicked, [this, i]() {
                m_loss = ActiveLoss(i);
                configChanged();
            });
        }
        optCardLay->addWidget(lossWell);

        // Optimizer selector
        auto* optLbl = new QLabel(QStringLiteral("OPTIMIZER"));
        optLbl->setStyleSheet(QStringLiteral("color:#94A3B8; font-size:10px; font-weight:700; letter-spacing:0.5px;"));
        optCardLay->addWidget(optLbl);

        auto* optWell = new QFrame;
        optWell->setStyleSheet(QStringLiteral("background:#080C14; border:1px solid #1E293B; border-radius:0px;"));
        auto* optWellLay = new QHBoxLayout(optWell);
        optWellLay->setContentsMargins(2, 2, 2, 2);
        optWellLay->setSpacing(2);
        for (int i = 0; i < 3; ++i) {
            QPushButton* b = new QPushButton(QString::fromLatin1(kOptLabels[i]));
            b->setCheckable(true);
            b->setCursor(Qt::PointingHandCursor);
            b->setFixedHeight(28);
            b->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
            b->setStyleSheet(QStringLiteral(
                "QPushButton { background:transparent; color:#94A3B8; border:none; border-radius:0px; font-size:11px; font-weight:600; }"
                "QPushButton:hover { background:rgba(255,255,255,0.06); color:#F1F5F9; }"
                "QPushButton:checked { background:#0284C7; color:#FFFFFF; border:1px solid #38BDF8; font-weight:bold; }"));
            optWellLay->addWidget(b);
            m_optBtns.push_back(b);
        }
        setExclusive(m_optBtns);
        for (int i = 0; i < 3; ++i) {
            connect(m_optBtns[(std::size_t)i], &QPushButton::clicked, [this, i]() {
                m_opt = ActiveOptimizer(i);
                applyOptimizerDefaults();
                configChanged();
            });
        }
        optCardLay->addWidget(optWell);

        m_muRow = new QWidget;
        auto* muLay = new QHBoxLayout(m_muRow);
        muLay->setContentsMargins(0, 0, 0, 0);
        muLay->setSpacing(8);
        auto* muLbl = new QLabel(QStringLiteral("Momentum"));
        muLbl->setStyleSheet(QStringLiteral("color:#94A3B8; font-size:11px; font-weight:600;"));
        muLay->addWidget(muLbl);
        m_muSpin = new QDoubleSpinBox;
        m_muSpin->setRange(0.0, 0.999);
        m_muSpin->setDecimals(3);
        m_muSpin->setSingleStep(0.05);
        m_muSpin->setValue(0.9);
        connect(m_muSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [this](double v) {
            m_momentum = v;
            configChanged();
        });
        muLay->addWidget(m_muSpin, 1);
        optCardLay->addWidget(m_muRow);

        m_adamRow = new QWidget;
        auto* adLay = new QHBoxLayout(m_adamRow);
        adLay->setContentsMargins(0, 0, 0, 0);
        adLay->setSpacing(8);
        auto* b1Col = new QVBoxLayout;
        b1Col->setSpacing(3);
        auto* b1Lbl = new QLabel(QStringLiteral("β1"));
        b1Lbl->setStyleSheet(QStringLiteral("color:#94A3B8; font-size:10px; font-weight:700;"));
        b1Col->addWidget(b1Lbl);
        m_b1Spin = new QDoubleSpinBox;
        m_b1Spin->setRange(0.5, 0.9999);
        m_b1Spin->setDecimals(4);
        m_b1Spin->setValue(0.9);
        connect(m_b1Spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [this](double v) {
            m_beta1 = v;
            configChanged();
        });
        b1Col->addWidget(m_b1Spin);
        adLay->addLayout(b1Col);

        auto* b2Col = new QVBoxLayout;
        b2Col->setSpacing(3);
        auto* b2Lbl = new QLabel(QStringLiteral("β2"));
        b2Lbl->setStyleSheet(QStringLiteral("color:#94A3B8; font-size:10px; font-weight:700;"));
        b2Col->addWidget(b2Lbl);
        m_b2Spin = new QDoubleSpinBox;
        m_b2Spin->setRange(0.9, 0.99999);
        m_b2Spin->setDecimals(5);
        m_b2Spin->setValue(0.999);
        connect(m_b2Spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [this](double v) {
            m_beta2 = v;
            configChanged();
        });
        b2Col->addWidget(m_b2Spin);
        adLay->addLayout(b2Col);

        auto* epsCol = new QVBoxLayout;
        epsCol->setSpacing(3);
        auto* epsLbl = new QLabel(QStringLiteral("ε"));
        epsLbl->setStyleSheet(QStringLiteral("color:#94A3B8; font-size:10px; font-weight:700;"));
        epsCol->addWidget(epsLbl);
        m_epsCombo = new QComboBox;
        m_epsCombo->addItems({QStringLiteral("1e-6"), QStringLiteral("1e-7"),
                              QStringLiteral("1e-8"), QStringLiteral("1e-9")});
        m_epsCombo->setCurrentIndex(2);
        connect(m_epsCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int i) {
            m_eps = std::pow(10.0, -6 - i);
            configChanged();
        });
        epsCol->addWidget(m_epsCombo);
        adLay->addLayout(epsCol);

        optCardLay->addWidget(m_adamRow);

        s4->addWidget(optCard);

        // Warnings live outside the collapsed box so they are never hidden.
        m_warnLbl = new QLabel;
        m_warnLbl->setWordWrap(true);
        m_warnLbl->setStyleSheet(QStringLiteral("color:#F5A623;"));
        lv->addWidget(m_warnLbl);

        // 5. run status (§42 summary lives here)
        auto* sec5 = new CollapsibleSection(QStringLiteral("5. RUN STATUS"), true, m_configBox);
        lv->addWidget(sec5);
        QVBoxLayout* s5 = sec5->content();
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
        s5->addWidget(panel);

        // (No dead stretch here: the scroll area above takes all extra space,
        // keeping SAVE/LOAD + START docked at the bottom.)
        auto* modelRow = new QHBoxLayout;
        modelRow->setSpacing(8);
        m_saveBtn = makeBtn(QStringLiteral("SAVE MODEL"));
        m_loadBtn = makeBtn(QStringLiteral("LOAD MODEL"));
        m_reportBtn = makeBtn(QStringLiteral("EXPORT REPORT"));
        m_saveBtn->setEnabled(false);
        m_reportBtn->setEnabled(false);
        m_reportBtn->setToolTip(QStringLiteral("Generate standalone HTML/SVG experiment report and open in web browser"));
        connect(m_saveBtn, &QPushButton::clicked, [this]() { onSaveModel(); });
        connect(m_loadBtn, &QPushButton::clicked, [this]() { onLoadModel(); });
        connect(m_reportBtn, &QPushButton::clicked, [this]() { onExportReport(); });
        modelRow->addWidget(m_saveBtn, 1);
        modelRow->addWidget(m_loadBtn, 1);
        modelRow->addWidget(m_reportBtn, 1);
        outer->addLayout(modelRow);

        // Dedicated pacing & step control row (clean spacing, responsive controls)
        auto* paceRow = new QHBoxLayout;
        paceRow->setSpacing(8);
        auto* paceLbl = new QLabel(QStringLiteral("Pacing:"));
        paceLbl->setStyleSheet(QStringLiteral("color:#94A3B8; font-size:11px; font-weight:bold;"));
        m_speedSlider = new QSlider(Qt::Horizontal);
        m_speedSlider->setRange(0, 20);
        m_speedSlider->setValue(0);
        m_speedSlider->setToolTip("Pacing delay per epoch (slide right to slow down for visual observation)");
        connect(m_speedSlider, &QSlider::valueChanged, [](int v) {
            g_bridge.epochDelayMs.store(v * 2); // 0 to 40ms delay per epoch
        });
        m_stepBtn = makeBtn(QStringLiteral("Step 10 Ep"), 90);
        m_stepBtn->setToolTip("Advance 10 epochs incrementally");
        connect(m_stepBtn, &QPushButton::clicked, [this]() {
            if (g_bridge.isTraining) return;
            ExperimentConfig cfg = currentConfig();
            cfg.epochs = 10;
            g_bridge.resetLive();
            m_resultsShownFor = false;
            m_seenSeq = g_bridge.runSeq;
            g_bridge.isTraining = true;
            g_bridge.stopRequested = false;
            g_trainThread = std::make_unique<std::thread>([cfg]() {
                ExperimentResult res = ExperimentController::run(cfg, &g_bridge, &g_bridge.stopRequested);
                std::lock_guard<std::mutex> lock(g_bridge.mtx);
                if (!res.error.empty()) {
                    g_bridge.lastError = std::string("Error: ") + res.error;
                } else {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                                  "Train loss %.4f | Train acc %.1f%% | Test acc %.1f%% (%.2fs)%s",
                                  res.trainLoss, res.trainAcc * 100.0, res.testAcc * 100.0,
                                  res.seconds, res.stopped ? " — stopped" : "");
                    g_bridge.summary = buf;
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
                    g_bridge.lastWarning = res.warning;
                    g_bridge.lastConfusion = res.confusion;
                    g_bridge.lastNumClasses = res.numClasses;
                    g_bridge.lastHasConfusion = res.hasConfusion;
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
                }
                g_bridge.isTraining = false;
            });
        });
        paceRow->addWidget(paceLbl);
        paceRow->addWidget(m_speedSlider, 1);
        paceRow->addWidget(m_stepBtn);
        outer->addLayout(paceRow);
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
        setExclusive({m_tabLoss, m_tabAcc, m_tabBnd, m_tabNet});
        m_tabLoss->setChecked(true);
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
            "QMainWindow, QWidget#qt_top { background:#05070B; }"
            "QFrame#topbar { background:#0A0D15; border:none; border-bottom:1px solid #1E293B; }"
            "QFrame#card { background:#0A0E17; border:1px solid #1E293B; border-radius:0px; }"
            "QFrame#panel { background:#0D121E; border:1px solid #1E293B; border-radius:0px; }"
            "QLabel { color:#E2E8F0; }"
            "QPushButton { background:#0F1420; color:#E2E8F0; border:1px solid #1E293B; border-radius:0px; padding:6px 12px; font-weight:500; }"
            "QPushButton:hover { background:#1E293B; border:1px solid #38BDF8; }"
            "QPushButton:checked { background:#0284C7; color:#FFFFFF; border:1px solid #38BDF8; font-weight:bold; }"
            "QPushButton:disabled { background:#080B12; color:#475569; border:1px solid #151C2A; }"
            "QPushButton#trainBtn { background:qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #10B981, stop:1 #059669); color:#FFFFFF; border:1px solid #059669; border-radius:0px; font-weight:bold; }"
            "QPushButton#trainBtn:hover { background:#10B981; border:1px solid #34D399; }"
            "QPushButton#stopBtn { background:qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #EF4444, stop:1 #B91C1C); color:#FFFFFF; border:1px solid #B91C1C; border-radius:0px; font-weight:bold; }"
            "QPushButton#stopBtn:hover { background:#DC2626; border:1px solid #F87171; }"
            "QSlider::groove:horizontal { background:#151C2A; height:4px; border-radius:0px; }"
            "QSlider::handle:horizontal { background:#38BDF8; border:1px solid #0284C7; width:12px; height:12px; margin:-4px 0; border-radius:0px; }"
            "QSlider::sub-page:horizontal { background:#0284C7; border-radius:0px; }"
            "QSpinBox, QComboBox, QDoubleSpinBox { background:#0F1420; color:#F1F5F9; border:1px solid #1E293B; border-radius:0px; padding:4px 8px; }"
            "QComboBox QAbstractItemView { background:#0F1420; color:#F1F5F9; selection-background-color:#0284C7; border:1px solid #1E293B; }"
            "QCheckBox { color:#94A3B8; }"
            "QStatusBar { background:#05070B; color:#64748B; border-top:1px solid #1E293B; }"
            "QToolTip { background:#0A0E17; color:#F8FAFC; border:1px solid #1E293B; border-radius:0px; padding:6px; font-size:11px; }"
            "QTableWidget { background:#0F1420; color:#F8FAFC; gridline-color:#1E293B; border:1px solid #1E293B; border-radius:0px; }"
            "QScrollArea { background:transparent; border:none; }"
            "QScrollBar:vertical { background:transparent; width:8px; margin:0; border:none; }"
            "QScrollBar::handle:vertical { background:#1E293B; min-height:24px; border-radius:0px; }"
            "QScrollBar::handle:vertical:hover { background:#334155; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; border:none; }"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:none; border:none; }"
            "QHeaderView::section { background:#0F1420; color:#94A3B8; border:1px solid #1E293B; padding:5px; }"
            "QDialog { background:#0A0E17; border:1px solid #1E293B; }"
        ));
    }

    // ---- config pipeline: preview -> validate -> READY/ERROR ----
    // Popup editor for 1-8 hidden layers (keeps the left panel compact).
    void openHiddenDialog() {
        if (g_bridge.isTraining) return; // config locked during training
        auto* dlg = new QDialog(this);
        dlg->setWindowTitle(QStringLiteral("Hidden layers"));
        dlg->setAttribute(Qt::WA_DeleteOnClose);
        dlg->setMinimumWidth(430);
        auto* lay = new QVBoxLayout(dlg);
        auto* hint = new QLabel(QString::asprintf("1-%d hidden layers, 1-64 neurons each.", kMaxHidden));
        hint->setStyleSheet(QStringLiteral("color:#8A90A0;"));
        lay->addWidget(hint);
        auto* box = new QWidget;
        auto* rows = new QVBoxLayout(box);
        rows->setContentsMargins(0, 0, 0, 0);
        rows->setSpacing(4);
        lay->addWidget(box);

        struct HDlgState {
            std::vector<HiddenCfg> work;
            std::vector<QWidget*> rows;
            QVBoxLayout* layout = nullptr;
            std::function<void()> rebuild;
        };
        auto st = std::make_shared<HDlgState>();
        st->work = m_hidden;
        st->layout = rows;
        st->rebuild = [st]() {
            for (QWidget* w : st->rows) {
                st->layout->removeWidget(w);
                w->deleteLater();
            }
            st->rows.clear();
            for (std::size_t i = 0; i < st->work.size(); ++i) {
                auto* row = new QWidget;
                auto* rl = new QHBoxLayout(row);
                rl->setContentsMargins(0, 0, 0, 0);
                rl->setSpacing(4);
                auto* nm = new QLabel(QString::asprintf("H%llu", (unsigned long long)i + 1));
                nm->setStyleSheet(QStringLiteral("color:#C9CDD8;"));
                nm->setFixedWidth(32);
                rl->addWidget(nm);
                auto* sz = new QSpinBox;
                sz->setRange(1, 64);
                sz->setValue(st->work[i].n);
                sz->setFixedWidth(60);
                connect(sz, &QSpinBox::valueChanged, [st, i](int v) { st->work[i].n = v; });
                rl->addWidget(sz);
                auto* ac = new QComboBox;
                for (auto lbl : kActLabels) ac->addItem(QString::fromLatin1(lbl));
                ac->setCurrentIndex((int)st->work[i].act);
                connect(ac, QOverload<int>::of(&QComboBox::currentIndexChanged),
                        [st, i](int a) { st->work[i].act = ActiveActivation(std::max(0, std::min(4, a))); });
                rl->addWidget(ac, 1);
                auto* up = new QPushButton(QStringLiteral("^"));
                up->setFixedWidth(30);
                up->setToolTip(QStringLiteral("Move up"));
                up->setEnabled(i > 0);
                connect(up, &QPushButton::clicked, [st, i]() {
                    std::swap(st->work[i], st->work[i - 1]);
                    st->rebuild();
                });
                rl->addWidget(up);
                auto* dn = new QPushButton(QStringLiteral("v"));
                dn->setFixedWidth(30);
                dn->setToolTip(QStringLiteral("Move down"));
                dn->setEnabled(i + 1 < st->work.size());
                connect(dn, &QPushButton::clicked, [st, i]() {
                    std::swap(st->work[i], st->work[i + 1]);
                    st->rebuild();
                });
                rl->addWidget(dn);
                auto* del = new QPushButton(QStringLiteral("x"));
                del->setFixedWidth(30);
                del->setToolTip(QStringLiteral("Remove layer"));
                del->setEnabled(st->work.size() > 1);
                connect(del, &QPushButton::clicked, [st, i]() {
                    st->work.erase(st->work.begin() + (std::ptrdiff_t)i);
                    st->rebuild();
                });
                rl->addWidget(del);
                st->layout->addWidget(row);
                st->rows.push_back(row);
            }
        };
        st->rebuild();
        // Break the rebuild self-cycle when the dialog closes.
        connect(dlg, &QDialog::finished, [st]() { st->rebuild = nullptr; });
        auto* addBtn = new QPushButton(QStringLiteral("+ Add hidden layer"));
        connect(addBtn, &QPushButton::clicked, [st]() {
            if ((int)st->work.size() < kMaxHidden) {
                st->work.push_back({8, ACT_RELU});
                st->rebuild();
            }
        });
        lay->addWidget(addBtn);
        auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
        connect(buttons, &QDialogButtonBox::accepted, this, [this, dlg, st]() {
            if (!st->work.empty()) {
                m_hidden = st->work;
                configChanged();
            }
            dlg->accept();
        });
        connect(buttons, &QDialogButtonBox::rejected, dlg, &QDialog::reject);
        lay->addWidget(buttons);
        dlg->show();
    }

    void configChanged() {
        if (g_bridge.isTraining) return; // config locked during training
        g_bridge.resetLive();
        m_resultsShownFor = false;
        refreshConfigUi();
    }

    // Skeleton preview of the CURRENT config (dims from data preview).
    // Called on every config change so the Network tab can never show a
    // stale trained net as if it were the current architecture.
    void refreshNetPreview() {
        if (g_bridge.isTraining) return;
        std::string fp = archSummary() + "|" + m_previewFp;
        if (fp == m_shownNetFp) return;
        m_shownNetFp = fp;
        if (!m_previewErr.empty()) {
            m_netview->setArch(ArchDesc{});
            return;
        }
        ArchDesc ad;
        ad.valid = true;
        auto shortAct = [](const std::string& n) -> QString {
            if (n == "sigmoid") return QStringLiteral("Sig");
            if (n == "tanh") return QStringLiteral("Tanh");
            if (n == "relu") return QStringLiteral("ReLU");
            if (n == "leaky_relu") return QStringLiteral("L-ReLU");
            if (n == "swish") return QStringLiteral("Swish");
            return QStringLiteral("Lin");
        };
        ad.sizes.push_back(m_preview.inDim);
        ad.names.push_back(QStringLiteral("Input"));
        ExperimentConfig cfg = currentConfig();
        for (std::size_t i = 0; i < cfg.hidden.size(); ++i) {
            ad.sizes.push_back(std::size_t(cfg.hidden[i]));
            ad.names.push_back(QString::asprintf("H%llu·", (unsigned long long)i + 1) +
                               shortAct(cfg.hiddenActs[i]));
        }
        ad.sizes.push_back(m_preview.outDim);
        ad.names.push_back(QStringLiteral("Output·") + shortAct(cfg.outputAct));
        m_netview->setArch(std::move(ad));
    }

    // Never throws into Qt: any unexpected failure becomes an error label
    // (an exception escaping a slot would terminate the whole app).
    void refreshConfigUi() try {
        checkOnly(m_dsBtns, (int)m_dataset);
        checkOnly(m_lossBtns, (int)m_loss);
        checkOnly(m_optBtns, (int)m_opt);
        // Preview FIRST: arch summary + info panel below consume it.
        refreshPreview();
        m_muRow->setVisible(m_opt == OPT_MOMENTUM);
        m_adamRow->setVisible(m_opt == OPT_ADAM);
        m_csvRow->setVisible(m_dataset == DS_CSV);
        m_csvPathLbl->setText(m_csvPath.empty() ? QStringLiteral("No file — drag and drop a .csv file onto the window.")
                                                : QString::fromStdString(m_csvPath));
        m_layersLbl->setText(QString::asprintf("LAYERS: %d (1-%d)", (int)m_hidden.size(), kMaxHidden));
        m_archLbl->setText(QString::fromStdString(archSummary()));
        {
            QStringList parts;
            for (std::size_t i = 0; i < m_hidden.size(); ++i)
                parts.push_back(QString::asprintf("H%llu:%d %s", (unsigned long long)i + 1,
                    m_hidden[i].n, kActLabels[(int)m_hidden[i].act]));
            m_hiddenSumLbl->setText(parts.join(QStringLiteral("  ·  ")));
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

        // (Preview was already refreshed above; its fingerprint cache makes
        // a second call here a no-op, so don't pay for it twice.)
        if (!m_previewErr.empty()) {
            m_state = ST_ERROR;
            m_dsInfoLbl->setText(QString::fromStdString(std::string("Dataset error:\n") + m_previewErr));
            m_warnLbl->setText(QString());
            m_cfgLbl->setText(QStringLiteral("Fix the dataset configuration to continue."));
            m_trainBtn->setEnabled(false);
            refreshNetPreview();
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
            if (d.tiny)
                info += QString::asprintf("Train: %llu | Val: — | Test: %llu (full table: tiny dataset)\n",
                                          (unsigned long long)d.nTrain, (unsigned long long)d.nTest);
            else
                info += QString::asprintf("Train: %llu | Val: %llu | Test: %llu (split 80/10/10)\n",
                                          (unsigned long long)d.nTrain, (unsigned long long)d.nVal,
                                          (unsigned long long)d.nTest);
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
                + QString::asprintf(" (eff %llu) | seed %d%s | %s", (unsigned long long)eff, m_seed,
                                    m_shuffle ? "" : " noshuffle", normN));
        }
        refreshNetPreview();
        refreshLiveUi();
    } catch (const std::exception& e) {
        qWarning("refreshConfigUi: %s", e.what());
        m_state = ST_ERROR;
        m_dsInfoLbl->setText(QStringLiteral("Internal error:\n") +
                             QString::fromStdString(e.what()));
        m_trainBtn->setEnabled(false);
    } catch (...) {
        qWarning("refreshConfigUi: unknown error");
        m_state = ST_ERROR;
        m_trainBtn->setEnabled(false);
    }

    void syncTabs() {
        checkOnly({m_tabLoss, m_tabAcc, m_tabBnd, m_tabNet}, (int)m_view);
        m_plot->setVisible(m_view == VIEW_LOSS || m_view == VIEW_ACC);
        m_boundary->setVisible(m_view == VIEW_BOUNDARY);
        m_netview->setVisible(m_view == VIEW_NETWORK);
        refreshLiveUi();
    }

    static QString badgeStyle(const char* bg, const char* fg) {
        return QString::asprintf("background:%s; color:%s; border-radius:0px; border:1px solid rgba(255,255,255,0.15); font-weight:bold;", bg, fg);
    }

    // Per-tick change caches: refreshLiveUi runs at 10 Hz, and touching a
    // widget (setText/setStyleSheet/setObjectName/setEnabled) forces a
    // repolish + layout pass. Suppressing identical updates keeps the window
    // out of a perpetual re-layout storm that tears frames into dotted text.
    QString m_cEpoch, m_cLoss, m_cTrAcc, m_cValAcc, m_cTeAcc, m_cProg;
    QString m_cSum, m_cSumBar, m_cSumStyle, m_cTrainTxt, m_cTrainObj;
    int m_cBadge = -1, m_cCfgEn = -1, m_cSaveEn = -1, m_cTrainEn = -1;
    // Plot repaint gate: history vectors only ever append, so equal
    // lengths + same view + same epoch means pixel-identical output.
    // (Init to impossible values so the very first tick always paints.)
    std::size_t m_cPlotA = (std::size_t)-1, m_cPlotB = (std::size_t)-1;
    int m_cPlotV = -1, m_cPlotE = -1;

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
            if (m_reportBtn) m_reportBtn->setEnabled(en);
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
        bool fHasVal = false;
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
            setOnce(m_testAccLbl, m_cTeAcc, QString::asprintf("Test Acc: %.1f%%", fTeA * 100.0f));
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

        // Copying multi-hundred-thousand-point curves and repainting on
        // every 100 ms tick — including fully idle windows — caused the
        // flicker the caches above were built to prevent. Repaint only when
        // new epochs actually streamed in (or the view switched).
        std::size_t keyA = trainLoss.size() + trainAcc.size();
        std::size_t keyB = valLoss.size() + valAcc.size();
        if (keyA != m_cPlotA || keyB != m_cPlotB || m_view != m_cPlotV || liveEpoch != m_cPlotE) {
            m_cPlotA = keyA;
            m_cPlotB = keyB;
            m_cPlotV = m_view;
            m_cPlotE = liveEpoch;
            m_plot->setCurves(m_view == VIEW_ACC ? trainAcc : trainLoss,
                              m_view == VIEW_ACC ? valAcc : valLoss,
                              m_view == VIEW_ACC, !valLoss.empty() || !valAcc.empty(), liveEpoch);
        }
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
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                              "Train loss %.4f | Train acc %.1f%% | Test acc %.1f%% (%.2fs)%s",
                              res.trainLoss, res.trainAcc * 100.0, res.testAcc * 100.0,
                              res.seconds, res.stopped ? " — stopped" : "");
                g_bridge.summary = buf;
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
                g_bridge.lastWarning = res.warning;
                g_bridge.lastConfusion = res.confusion;
                g_bridge.lastNumClasses = res.numClasses;
                g_bridge.lastHasConfusion = res.hasConfusion;
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
            m_boundary->setPredictor(net);
        } else {
            m_boundary->setData(QImage(), {}, {}, {}, {}, 0, 1, 0, 1, false);
            m_boundary->setPredictor(nullptr);
        }
        // Network diagram: trained net with live firing (replaces any preview).
        if (net && net->numLayers() > 0) {
            Vector probe;
            if (tr.size() > 0) probe = tr.input(0);
            else if (te.size() > 0) probe = te.input(0);
            QString tag = QStringLiteral("TRAINED — ") +
                QString::fromStdString(describeNet(*net));
            m_netview->setNet(net, probe, tag);
            m_shownNetFp = archSummary() + "|" + m_previewFp;
        } else {
            m_netview->setArch(ArchDesc{});
        }
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
        row("Test Loss", QString::asprintf("%.4f", g_bridge.lastTestLoss));
        row("Test Acc", QString::asprintf("%.1f%%", g_bridge.lastTestAcc * 100.0));
        if (g_bridge.lastTiny)
            row("Note", QStringLiteral("Tiny dataset: test == train set."));
        lay->addLayout(form);
        if (g_bridge.lastHasConfusion && g_bridge.lastNumClasses >= 2 && g_bridge.lastNumClasses <= 10) {
            auto* cmTitle = new QLabel(QStringLiteral("Confusion matrix (rows = actual, cols = predicted)"));
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

    void onExportReport() {
        std::shared_ptr<NeuralNetwork> net;
        Dataset trainData;
        TrainingHistory hist;
        std::string optName, lossName, dsDesc;
        int epochs = 0;
        double finalLoss = 0.0, finalAcc = 0.0, lr = 0.01;

        {
            std::lock_guard<std::mutex> lock(g_bridge.mtx);
            if (!g_bridge.hasResult || !g_bridge.lastNet) {
                QMessageBox::information(this, QStringLiteral("Export Report"),
                    QStringLiteral("Please complete at least one training run before exporting a report."));
                return;
            }
            net = g_bridge.lastNet;
            trainData = g_bridge.lastTrain;
            optName = g_bridge.lastOptDesc;
            lossName = g_bridge.lastLoss;
            dsDesc = g_bridge.lastDsDesc;
            epochs = g_bridge.lastEpochsRun;
            finalLoss = g_bridge.lastTrainLoss;
            finalAcc = g_bridge.lastTrainAcc;
            for (float v : g_bridge.trainLoss) hist.trainLoss.push_back((double)v);
            for (float v : g_bridge.valLoss) hist.validationLoss.push_back((double)v);
            for (float v : g_bridge.trainAcc) hist.trainAcc.push_back((double)v);
            for (float v : g_bridge.valAcc) hist.validationAcc.push_back((double)v);
        }

        try {
            NetworkGraph ng(*net);
            WeightsTable wt(*net);
            HtmlReport report = HtmlReportBuilder(std::string("MiniANN Workbench Report: ") + (dsDesc.empty() ? "Model" : dsDesc))
                                    .withSeries(optName.empty() ? "optimizer" : optName, hist)
                                    .withBoundary(*net, trainData)
                                    .withPre("network structure", ng.render())
                                    .withPre("learned weights", wt.render())
                                    .withSummary(finalLoss, finalAcc, epochs, optName, lr)
                                    .build();
            std::string filename = "workbench_report.html";
            report.save(filename);
            QFileInfo fi(QString::fromStdString(filename));
            QUrl fileUrl = QUrl::fromLocalFile(fi.absoluteFilePath());
            QDesktopServices::openUrl(fileUrl);
        } catch (const std::exception& e) {
            QMessageBox::warning(this, QStringLiteral("Export Report Error"),
                QString::fromStdString(std::string("Could not generate report:\n") + e.what()));
        }
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
            if (net.numLayers() < 2 || net.numLayers() > std::size_t(kMaxHidden + 1)) {
                QMessageBox::warning(this, QStringLiteral("Load model"),
                                     QString::asprintf("Model has %llu hidden layers; the editor supports 1-%d.",
                                                       (unsigned long long)net.numLayers() - 1, kMaxHidden));
                return;
            }
            std::size_t nh = net.numLayers() - 1;
            std::vector<HiddenCfg> loaded;
            for (std::size_t l = 0; l < nh; ++l) {
                HiddenCfg h;
                h.n = (int)net.layers()[l].size();
                std::string an = net.layers()[l].neurons().empty()
                    ? "" : net.layers()[l].neurons()[0].activation().name();
                try {
                    h.act = actFromName(an);
                } catch (const std::exception&) {
                    QMessageBox::warning(this, QStringLiteral("Load model"),
                                         QString::fromStdString("Unsupported hidden activation '" + an + "'."));
                    return;
                }
                loaded.push_back(h);
            }
            m_hidden = loaded;
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
            auto shown = std::make_shared<NeuralNetwork>(net);
            QString tag = QStringLiteral("LOADED model — ") +
                QString::fromStdString(describeNet(net));
            m_netview->setNet(shown, Vector{},
                              tag + QStringLiteral(" — train to replace"));
            m_shownNetFp = archSummary() + "|" + m_previewFp + "|loaded";
            pushUiFromState();
            configChanged();
            // configChanged() repaints a skeleton preview; restore the loaded
            // net on top since its weights are worth inspecting as-is.
            m_netview->setNet(shown, Vector{}, tag);
            m_shownNetFp = archSummary() + "|" + m_previewFp + "|loaded";
            std::lock_guard<std::mutex> lock(g_bridge.mtx);
            g_bridge.summary = "Model loaded (" + describeNet(net) + ") — pick a matching dataset, then train or inspect Network.";
        } catch (const std::exception& e) {
            QMessageBox::warning(this, QStringLiteral("Load model"),
                                 QString::fromStdString(std::string("Could not load model:\n") + e.what()));
        }
    }

    void pushUiFromState() {
        checkOnly(m_dsBtns, (int)m_dataset);
        checkOnly(m_lossBtns, (int)m_loss);
        checkOnly(m_optBtns, (int)m_opt);
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
        m_normCombo->blockSignals(true);
        m_normCombo->setCurrentIndex(m_normMode);
        m_normCombo->blockSignals(false);
        m_muRow->setVisible(m_opt == OPT_MOMENTUM);
        m_adamRow->setVisible(m_opt == OPT_ADAM);
    }

    void applyPresetXor(bool refresh) {
        m_dataset = DS_XOR;
        m_hidden = {{8, ACT_TANH}};
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
        // Mirrors demos/iris_demo.cpp (4-6-3 tanh/sigmoid, Adam 0.01, batch 8):
        // the proven-stable Iris setup. Deeper ReLU stacks collapsed to 33%
        // on some seeds with only 24 training rows.
        m_dataset = DS_IRIS;
        m_hidden = {{6, ACT_TANH}};
        m_outputAct = "sigmoid";
        m_loss = LOSS_CCE;
        m_opt = OPT_ADAM;
        m_lr = 0.01;
        m_epochsTarget = 1500;
        m_batchIdx = 1; // 8
        m_seed = 42;
        m_shuffle = true;
        m_normMode = 1;
        pushUiFromState();
        configChanged();
    }
    void applyPresetBinary() {
        m_dataset = DS_AND;
        m_hidden = {{4, ACT_TANH}};
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

public:
    void selectTab(const std::string& name) {
        if (name == "acc") { m_view = VIEW_ACC; syncTabs(); }
        else if (name == "boundary" || name == "bnd") { m_view = VIEW_BOUNDARY; syncTabs(); }
        else if (name == "network" || name == "net") { m_view = VIEW_NETWORK; syncTabs(); }
        else { m_view = VIEW_LOSS; syncTabs(); }
    }

    void scrollConfig(int val) {
        if (m_cfgScroll && m_cfgScroll->verticalScrollBar()) {
            m_cfgScroll->verticalScrollBar()->setValue(val);
        }
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
    QString shotPath, dumpPath, tabArg;
    int scrollArg = -1;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--shot" && i + 1 < argc) shotPath = QString::fromLocal8Bit(argv[++i]);
        if (std::string(argv[i]) == "--dump" && i + 1 < argc) dumpPath = QString::fromLocal8Bit(argv[++i]);
        if (std::string(argv[i]) == "--tab" && i + 1 < argc) tabArg = QString::fromLocal8Bit(argv[++i]);
        if (std::string(argv[i]) == "--scroll" && i + 1 < argc) scrollArg = std::atoi(argv[++i]);
    }
    if (!tabArg.isEmpty()) {
        w.selectTab(tabArg.toStdString());
    }
    if (scrollArg >= 0) {
        w.scrollConfig(scrollArg);
    }
    if (!shotPath.isEmpty() || !dumpPath.isEmpty()) {
        QTimer::singleShot(1500, [&, tabArg, scrollArg]() {
            if (!tabArg.isEmpty()) w.selectTab(tabArg.toStdString());
            if (scrollArg >= 0) w.scrollConfig(scrollArg);
            if (!dumpPath.isEmpty()) w.dumpDebug(dumpPath);
            if (!shotPath.isEmpty()) w.grab().save(shotPath);
            app.quit();
        });
    }
    int rc = app.exec();
    if (g_trainThread && g_trainThread->joinable()) g_trainThread->join();
    return rc;
}
