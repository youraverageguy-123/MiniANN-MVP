#pragma once
#include "miniann/types.hpp"
#include <stdexcept>
#include <string>
#include <vector>
#include <cstddef>

namespace miniann {

class IMetric {
public:
    virtual ~IMetric() = default;
    virtual double evaluate(const std::vector<Vector>& preds,
                            const std::vector<Vector>& targets) const = 0;
    virtual std::string name() const = 0;
};

class Accuracy : public IMetric {
public:
    explicit Accuracy(double threshold = 0.5) : threshold_(threshold) {
        if (!(threshold > 0.0) || !(threshold < 1.0))
            throw std::invalid_argument("Accuracy: threshold must be in (0,1)");
    }
    double evaluate(const std::vector<Vector>& preds,
                    const std::vector<Vector>& targets) const override;
    std::string name() const override { return "accuracy"; }
    double threshold() const { return threshold_; }
private:
    double threshold_ = 0.5;
};

class ConfusionMatrix : public IMetric {
public:
    explicit ConfusionMatrix(std::size_t numClasses, double threshold = 0.5)
        : numClasses_(numClasses), threshold_(threshold) {
        if (numClasses == 0) throw std::invalid_argument("ConfusionMatrix: numClasses must be > 0");
    }
    double evaluate(const std::vector<Vector>& preds,
                    const std::vector<Vector>& targets) const override;
    std::string name() const override { return "confusion_matrix"; }
    void print() const;
    std::string toString() const;
    double precision(std::size_t classIdx) const;
    double recall(std::size_t classIdx) const;
    double f1(std::size_t classIdx) const;
    double macroF1() const;
    // Raw counts after evaluate(): rows = actual, cols = predicted.
    const std::vector<std::vector<std::size_t>>& matrix() const { return matrix_; }
private:
    std::size_t numClasses_;
    double threshold_ = 0.5;
    mutable std::vector<std::vector<std::size_t>> matrix_;
};

} // namespace miniann
