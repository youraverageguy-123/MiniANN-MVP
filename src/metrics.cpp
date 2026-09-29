#include "miniann/metrics.hpp"
#include <stdexcept>
#include <iostream>
#include <algorithm>

namespace miniann {

static std::size_t argmax(const Vector& v) {
    return std::size_t(std::max_element(v.begin(), v.end()) - v.begin());
}

double Accuracy::evaluate(const std::vector<Vector>& preds,
                          const std::vector<Vector>& targets) const {
    if (preds.size() != targets.size())
        throw std::invalid_argument("Accuracy: size mismatch");
    if (preds.empty()) return 0.0;
    std::size_t correct = 0;
    for (std::size_t i = 0; i < preds.size(); ++i) {
        if (preds[i].size() != targets[i].size())
            throw std::invalid_argument("Accuracy: dim mismatch");
        bool ok;
        if (preds[i].size() == 1) {
            ok = ((preds[i][0] >= 0.5) ? 1 : 0) == ((targets[i][0] >= 0.5) ? 1 : 0);
        } else {
            ok = argmax(preds[i]) == argmax(targets[i]);
        }
        if (ok) ++correct;
    }
    return double(correct) / double(preds.size());
}

double ConfusionMatrix::evaluate(const std::vector<Vector>& preds,
                                 const std::vector<Vector>& targets) const {
    matrix_.assign(numClasses_, std::vector<std::size_t>(numClasses_, 0));
    if (preds.size() != targets.size())
        throw std::invalid_argument("ConfusionMatrix: size mismatch");
    auto toClass = [](const Vector& v) -> std::size_t {
        if (v.size() == 1) return v[0] >= 0.5 ? 1 : 0;
        return argmax(v);
    };
    for (std::size_t i = 0; i < preds.size(); ++i) {
        std::size_t t = toClass(targets[i]);
        std::size_t p = toClass(preds[i]);
        if (t >= numClasses_ || p >= numClasses_)
            throw std::invalid_argument("ConfusionMatrix: class index out of range");
        matrix_[t][p]++;
    }
    std::size_t trace = 0, total = 0;
    for (std::size_t i = 0; i < numClasses_; ++i)
        for (std::size_t j = 0; j < numClasses_; ++j) {
            total += matrix_[i][j];
            if (i == j) trace += matrix_[i][j];
        }
    return total ? double(trace) / double(total) : 0.0;
}

void ConfusionMatrix::print() const {
    for (std::size_t i = 0; i < numClasses_; ++i) {
        for (std::size_t j = 0; j < numClasses_; ++j) {
            if (j) std::cout << " ";
            std::cout << matrix_[i][j];
        }
        std::cout << "\n";
    }
}

double ConfusionMatrix::precision(std::size_t c) const {
    if (c >= numClasses_ || matrix_.empty()) return 0.0;
    std::size_t tp = matrix_[c][c], col = 0;
    for (std::size_t i = 0; i < numClasses_; ++i) col += matrix_[i][c];
    return col ? double(tp) / double(col) : 0.0;
}

double ConfusionMatrix::recall(std::size_t c) const {
    if (c >= numClasses_ || matrix_.empty()) return 0.0;
    std::size_t tp = matrix_[c][c], row = 0;
    for (std::size_t j = 0; j < numClasses_; ++j) row += matrix_[c][j];
    return row ? double(tp) / double(row) : 0.0;
}

} // namespace miniann
