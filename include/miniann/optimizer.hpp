#pragma once
#include "miniann/network.hpp"
#include <vector>
#include <cstddef>

namespace miniann {

class IOptimizer {
public:
    virtual ~IOptimizer() = default;
    virtual void step(NeuralNetwork& net, std::size_t batchSize) = 0;
};

class SGD : public IOptimizer {
private:
    double lr_;
public:
    explicit SGD(double learningRate) : lr_(learningRate) {}
    void step(NeuralNetwork& net, std::size_t batchSize) override;
};

class Adam : public IOptimizer {
private:
    double lr_, beta1_, beta2_, eps_;
    std::size_t t_ = 0;
    std::vector<std::vector<Vector>> m_w_, v_w_;
    std::vector<std::vector<double>> m_b_, v_b_;
public:
    explicit Adam(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8);
    void step(NeuralNetwork& net, std::size_t batchSize) override;
};

} // namespace miniann
