#pragma once
#include "miniann/network.hpp"
#include <vector>
#include <cstddef>
#include <memory>
#include <string>

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

// Classical momentum: velocity v <- mu*v - lr*(grad/B), params += v.
// Sits between SGD (mu=0) and Adam (adaptive per-weight rates).
class Momentum : public IOptimizer {
private:
    double lr_, mu_;
    std::vector<std::vector<Vector>> v_w_;
    std::vector<std::vector<double>> v_b_;
public:
    explicit Momentum(double lr = 0.01, double mu = 0.9);
    void step(NeuralNetwork& net, std::size_t batchSize) override;
};

// Hyperparameters exposed to the UI. Defaults match each class's ctor defaults.
struct OptimizerConfig {
    double learningRate = 0.01;
    double momentum = 0.9;      // Momentum only
    double beta1 = 0.9;         // Adam only
    double beta2 = 0.999;       // Adam only
    double epsilon = 1e-8;      // Adam only
};

// Factory so the user picks the descent method by name at runtime:
// "sgd" | "momentum" | "adam". Throws invalid_argument otherwise.
class OptimizerFactory {
public:
    static std::unique_ptr<IOptimizer> create(const std::string& name, double lr);
    static std::unique_ptr<IOptimizer> create(const std::string& name, const OptimizerConfig& cfg);
};

} // namespace miniann
