#pragma once
// Automatic 3D network layout: each layer occupies a plane perpendicular to
// X; neurons form a centered rows×cols grid in Y/Z. No manual positions —
// 8, 64, 128 or 256 neurons all work from the neuron count alone.
// Pure math, no GL context needed, unit-tested. Moc-free.
#include <QVector3D>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace net3d {

struct LayerLayout3D {
    std::vector<QVector3D> positions; // one per neuron, centered on origin
    int rows = 0;
    int cols = 0;
};

// layerSizes[0] = input dim, then one entry per weight layer (hidden + out).
inline std::vector<LayerLayout3D> buildLayout3D(const std::vector<std::size_t>& layerSizes,
                                               float layerSpacing = 4.0f,
                                               float neuronSpacing = 1.15f) {
    std::vector<LayerLayout3D> out;
    const float nL = (float)layerSizes.size();
    for (std::size_t li = 0; li < layerSizes.size(); ++li) {
        const std::size_t n = layerSizes[li];
        LayerLayout3D L;
        if (n == 0) {
            out.push_back(std::move(L));
            continue;
        }
        L.cols = (int)std::ceil(std::sqrt((double)n));
        L.rows = (int)std::ceil((double)n / (double)L.cols);
        const float x = ((float)li - (nL - 1.0f) * 0.5f) * layerSpacing;
        for (std::size_t i = 0; i < n; ++i) {
            const int r = (int)(i / (std::size_t)L.cols);
            const int c = (int)(i % (std::size_t)L.cols);
            const float y = ((float)r - (float)(L.rows - 1) * 0.5f) * neuronSpacing;
            const float z = ((float)c - (float)(L.cols - 1) * 0.5f) * neuronSpacing;
            L.positions.emplace_back(x, y, z);
        }
        out.push_back(std::move(L));
    }
    return out;
}

// Adaptive spacing: the largest grid is kept within ~30 world units so huge
// layers (784 inputs) stay compact while small nets keep airy spacing.
inline float adaptiveNeuronSpacing(std::size_t maxNeurons, float lo = 0.45f, float hi = 1.15f) {
    if (maxNeurons == 0) return hi;
    const double dim = std::ceil(std::sqrt((double)maxNeurons));
    return std::clamp((float)(30.0 / dim), lo, hi);
}

// Layer (X) spacing scales with neuron spacing and depth so deep nets never
// compress into each other.
inline float adaptiveLayerSpacing(float neuronSpacing, std::size_t nLayers) {
    float s = neuronSpacing * 3.5f;
    if (s < 4.0f) s = 4.0f;
    if (nLayers > 6) s += (float)(nLayers - 6) * 0.3f;
    return s;
}

// Trainable parameter count for "N params" readouts: Σ(n_in·n_out + n_out).
inline std::size_t countParams(const std::vector<std::size_t>& layerSizes) {
    std::size_t total = 0;
    for (std::size_t l = 1; l < layerSizes.size(); ++l)
        total += layerSizes[l - 1] * layerSizes[l] + layerSizes[l];
    return total;
}

// Largest scene dimension (for camera fit).
inline float layoutExtent(const std::vector<LayerLayout3D>& layout) {
    float lo_x = 0, hi_x = 0, lo_y = 0, hi_y = 0, lo_z = 0, hi_z = 0;
    bool any = false;
    for (const auto& L : layout)
        for (const auto& p : L.positions) {
            if (!any) {
                lo_x = hi_x = p.x();
                lo_y = hi_y = p.y();
                lo_z = hi_z = p.z();
                any = true;
            } else {
                if (p.x() < lo_x) lo_x = p.x();
                if (p.x() > hi_x) hi_x = p.x();
                if (p.y() < lo_y) lo_y = p.y();
                if (p.y() > hi_y) hi_y = p.y();
                if (p.z() < lo_z) lo_z = p.z();
                if (p.z() > hi_z) hi_z = p.z();
            }
        }
    if (!any) return 0.0f;
    float dx = hi_x - lo_x, dy = hi_y - lo_y, dz = hi_z - lo_z;
    float m = dx;
    if (dy > m) m = dy;
    if (dz > m) m = dz;
    return m;
}

} // namespace net3d
