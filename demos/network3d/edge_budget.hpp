#pragma once
// Edge-budget selection for large networks: which connections get drawn.
// Pure STL (no Qt, no GL) so the policy is unit-tested headless.
// Policy: forced edges (selected neuron, weight updates) are always kept;
// otherwise the threshold pass is kept up to `cap`; when the pass exceeds
// `overFactor * cap` (default overview), only the strongest `overviewFrac`
// (default top 10%) survive. Moc-free.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace net3d {

struct EdgeW {
    int l = 0; // weight layer (col l -> col l+1)
    int a = 0; // from-neuron in col l
    int b = 0; // to-neuron in col l+1
    double w = 0.0;
    bool forced = false; // selected-neuron or weight-update edge
};

inline std::vector<EdgeW> applyEdgeBudget(std::vector<EdgeW> in, std::size_t cap,
                                         double overviewFrac = 0.10, double overFactor = 4.0) {
    if (in.size() <= cap) return in;
    std::stable_partition(in.begin(), in.end(),
                          [](const EdgeW& e) { return e.forced; });
    std::size_t forced = 0;
    while (forced < in.size() && in[forced].forced) ++forced;
    std::size_t rest = in.size() - forced;
    std::size_t keep = 0;
    if (rest > (std::size_t)(overFactor * (double)cap)) {
        // Overview mode: strongest overviewFrac of the non-forced edges.
        keep = (std::size_t)(overviewFrac * (double)rest);
    } else if (forced < cap) {
        keep = cap - forced;
    } // else: forced set alone exceeds the cap; keep it all (still bounded:
      // selection fan-in/out is at most two layers' worth of edges).
    if (forced + keep < in.size()) {
        std::nth_element(in.begin() + (std::ptrdiff_t)forced,
                         in.begin() + (std::ptrdiff_t)(forced + keep), in.end(),
                         [](const EdgeW& x, const EdgeW& y) { return std::abs(x.w) > std::abs(y.w); });
        in.resize(forced + keep);
    }
    return in;
}

} // namespace net3d
