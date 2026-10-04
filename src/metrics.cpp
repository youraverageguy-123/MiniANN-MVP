#include "miniann/metrics.hpp"
#include <sstream>
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
            ok = ((preds[i][0] >= threshold_) ? 1 : 0) == ((targets[i][0] >= 0.5) ? 1 : 0);
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
    double th = threshold_;
    auto toClass = [th](const Vector& v) -> std::size_t {
        if (v.size() == 1) return v[0] >= th ? 1 : 0;
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
    std::cout << toString();
}

std::string ConfusionMatrix::toString() const {
    std::ostringstream os;
    for (std::size_t i = 0; i < numClasses_; ++i) {
        for (std::size_t j = 0; j < numClasses_; ++j) {
            if (j) os << " ";
            os << matrix_[i][j];
        }
        os << "\n";
    }
    return os.str();
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

double ConfusionMatrix::f1(std::size_t c) const {
    double p = precision(c), r = recall(c);
    return (p + r) > 0.0 ? 2.0 * p * r / (p + r) : 0.0;
}

double ConfusionMatrix::macroF1() const {
    if (numClasses_ == 0 || matrix_.empty()) return 0.0;
    double s = 0.0;
    for (std::size_t c = 0; c < numClasses_; ++c) s += f1(c);
    return s / double(numClasses_);
}

} // namespace miniann
