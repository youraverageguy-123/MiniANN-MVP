#pragma once
// OOP visualization: every visual is an IVisualizer with render() -> string.
// Demos hold vectors of them and print; no plotting library required.
#include "miniann/network.hpp"
#include "miniann/trainer.hpp"
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace miniann {

class IVisualizer {
public:
    virtual ~IVisualizer() = default;
    virtual std::string render() = 0;
    virtual std::string title() const = 0;
};

// Downsampled loss line plot with axes, tick labels, log-y when span > 50x.
class LossCurve : public IVisualizer {
public:
    LossCurve(const TrainingHistory& h, int width = 60, int height = 10)
        : hist_(h), width_(width), height_(height) {}
    std::string render() override;
    std::string title() const override { return "loss curve"; }
private:
    const TrainingHistory& hist_;
    int width_, height_;
};

// 2-D decision map for 2-input / 1-output nets: '#' = >=0.5, '.' = <0.5,
// with training points overlaid (1 shown as '1', 0 shown as '0').
class BoundaryMap : public IVisualizer {
public:
    BoundaryMap(NeuralNetwork& net, const Dataset& data,
                int width = 41, int height = 21);
    std::string render() override;
    std::string title() const override { return "decision boundary (x1 horizontal, x2 vertical)"; }
private:
    NeuralNetwork& net_;
    const Dataset& data_;
    int width_, height_;
};

// Accuracy curve on the fixed 0..1 scale (same downsampling as LossCurve).
class AccuracyCurve : public IVisualizer {
public:
    AccuracyCurve(const TrainingHistory& h, int width = 60, int height = 10)
        : hist_(h), width_(width), height_(height) {}
    std::string render() override;
    std::string title() const override { return "accuracy curve"; }
private:
    const TrainingHistory& hist_;
    int width_, height_;
};

// Live training monitor: implements TrainingCallback so the console redraws
// loss + accuracy line plots and per-neuron firing bars while epochs stream
// in (TensorBoard-style, zero dependencies). Needs an ANSI-capable terminal
// (Windows Terminal / PowerShell); pass --no-live otherwise.
class LiveConsole : public TrainingCallback {
public:
    // probe: fixed input whose neuron outputs are shown firing each redraw.
    LiveConsole(NeuralNetwork& net, Vector probe, std::string title,
                int totalEpochs, int stride = 50);
    void onEpoch(int epoch, const TrainingHistory& hist) override;
private:
    void draw(int epoch, const TrainingHistory& hist);
    NeuralNetwork& net_;
    Vector probe_;
    std::string title_;
    int total_, stride_;
};

// One-line-per-layer architecture graph: dims, neuron counts, activations.
class NetworkGraph : public IVisualizer {
public:
    explicit NetworkGraph(const NeuralNetwork& net) : net_(net) {}
    std::string render() override;
    std::string title() const override { return "network structure"; }
private:
    const NeuralNetwork& net_;
};

// Learned parameters, one row per neuron: weights + bias.
class WeightsTable : public IVisualizer {
public:
    explicit WeightsTable(const NeuralNetwork& net) : net_(net) {}
    std::string render() override;
    std::string title() const override { return "learned weights"; }
private:
    const NeuralNetwork& net_;
};

// One plotted run: loss polyline (+ optional accuracy polyline) with a color.
struct ReportSeries {
    std::string name;
    std::string color; // "" = auto palette
    std::vector<double> loss;
    std::vector<double> acc;
};

// Self-contained HTML report: SVG loss chart (log-y) + SVG accuracy chart +
// SVG decision heatmap (2-D nets) + <pre> blocks. Open in any browser.
// Still dependency-free: the C++ side only writes text.
class HtmlReport {
public:
    explicit HtmlReport(std::string title) : title_(std::move(title)) {}
    void addSeries(const ReportSeries& s) { series_.push_back(s); }
    void setBoundary(NeuralNetwork* net, const Dataset* data) { bnet_ = net; bdata_ = data; }
    void addPre(const std::string& heading, const std::string& text);
    void save(const std::string& path) const; // throws runtime_error
private:
    std::string title_;
    std::vector<ReportSeries> series_;
    NeuralNetwork* bnet_ = nullptr;
    const Dataset* bdata_ = nullptr;
    std::vector<std::pair<std::string, std::string>> pres_;
};

} // namespace miniann
