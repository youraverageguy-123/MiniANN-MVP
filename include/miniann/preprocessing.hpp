#pragma once
// Shared data helpers (header-only). Single home for code that was
// copy-pasted in iris_demo / playground / gui_qt.
// Both GUIs + demos should call these instead of re-implementing.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include "miniann/dataset.hpp"

namespace miniann {

// Max-abs normalize inputs (keeps targets). Returns a new Dataset.
inline Dataset normalizeMaxAbs(const Dataset& d) {
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

// Integer class label (target[0]) -> one-hot vector of dim k.
inline Dataset oneHotEncode(const Dataset& raw, int k) {
    if (k < 2) throw std::invalid_argument("oneHotEncode: k must be >= 2");
    Dataset d;
    for (std::size_t i = 0; i < raw.size(); ++i) {
        int c = int(raw.target(i)[0]);
        Vector t(std::size_t(k), 0.0);
        if (c >= 0 && c < k) t[std::size_t(c)] = 1.0;
        d.add(raw.input(i), t);
    }
    return d;
}

// First-line column count for CSV target-col guessing (-1 = last col).
inline int detectCsvColumnCount(const std::string& path) {
    std::ifstream f(path);
    if (!f) return 0;
    std::string line;
    if (!std::getline(f, line)) return 0;
    std::stringstream ss(line);
    std::string cell;
    int n = 0;
    while (std::getline(ss, cell, ',')) ++n;
    return n;
}

} // namespace miniann
