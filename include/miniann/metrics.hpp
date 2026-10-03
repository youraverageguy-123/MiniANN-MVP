#pragma once
#include "miniann/types.hpp"
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
    double evaluate(const std::vector<Vector>& preds,
                    const std::vector<Vector>& targets) const override;
    std::string name() const override { return "accuracy"; }
};

class ConfusionMatrix : public IMetric {
public:
    explicit ConfusionMatrix(std::size_t numClasses) : numClasses_(numClasses) {}
    double evaluate(const std::vector<Vector>& preds,
                    const std::vector<Vector>& targets) const override;
    std::string name() const override { return "confusion_matrix"; }
    void print() const;
    double precision(std::size_t classIdx) const;
    double recall(std::size_t classIdx) const;
    // Raw counts after evaluate(): rows = actual, cols = predicted.
    const std::vector<std::vector<std::size_t>>& matrix() const { return matrix_; }
private:
    std::size_t numClasses_;
    mutable std::vector<std::vector<std::size_t>> matrix_;
};

} // namespace miniann
