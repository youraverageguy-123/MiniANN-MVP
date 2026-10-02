#include "miniann/optimizer.hpp"
#include <cmath>
#include <stdexcept>

namespace miniann {

void SGD::step(NeuralNetwork& net, std::size_t batchSize) {
    if (batchSize == 0) batchSize = 1;
    double scale = -lr_ / double(batchSize);
    for (auto& layer : net.layers()) {
        for (auto& n : layer.neurons()) {
            Vector dW(n.gradWeights().size());
            for (std::size_t i = 0; i < dW.size(); ++i)
                dW[i] = scale * n.gradWeights()[i];
            n.applyStep(dW, scale * n.gradBias());
        }
    }
}

Adam::Adam(double lr, double b1, double b2, double eps)
    : lr_(lr), beta1_(b1), beta2_(b2), eps_(eps) {}

void Adam::step(NeuralNetwork& net, std::size_t batchSize) {
    if (batchSize == 0) batchSize = 1;
    ++t_;
    auto& layers = net.layers();
    // lazy init moment tables to match topology
    if (m_w_.size() != layers.size()) {
        m_w_.resize(layers.size());
        v_w_.resize(layers.size());
        m_b_.resize(layers.size());
        v_b_.resize(layers.size());
        for (std::size_t l = 0; l < layers.size(); ++l) {
            auto& neurons = layers[l].neurons();
            m_w_[l].resize(neurons.size());
            v_w_[l].resize(neurons.size());
            m_b_[l].assign(neurons.size(), 0.0);
            v_b_[l].assign(neurons.size(), 0.0);
            for (std::size_t j = 0; j < neurons.size(); ++j) {
                m_w_[l][j].assign(neurons[j].weights().size(), 0.0);
                v_w_[l][j].assign(neurons[j].weights().size(), 0.0);
            }
        }
    }
    double b1t = 1.0 - std::pow(beta1_, double(t_));
    double b2t = 1.0 - std::pow(beta2_, double(t_));
    for (std::size_t l = 0; l < layers.size(); ++l) {
        auto& neurons = layers[l].neurons();
        for (std::size_t j = 0; j < neurons.size(); ++j) {
            auto& n = neurons[j];
            Vector dW(n.weights().size());
            for (std::size_t i = 0; i < dW.size(); ++i) {
                double g = n.gradWeights()[i] / double(batchSize);
                m_w_[l][j][i] = beta1_ * m_w_[l][j][i] + (1.0 - beta1_) * g;
                v_w_[l][j][i] = beta2_ * v_w_[l][j][i] + (1.0 - beta2_) * g * g;
                double mhat = m_w_[l][j][i] / b1t;
                double vhat = v_w_[l][j][i] / b2t;
                dW[i] = -lr_ * mhat / (std::sqrt(vhat) + eps_);
            }
            double gb = n.gradBias() / double(batchSize);
            m_b_[l][j] = beta1_ * m_b_[l][j] + (1.0 - beta1_) * gb;
            v_b_[l][j] = beta2_ * v_b_[l][j] + (1.0 - beta2_) * gb * gb;
            double mhat_b = m_b_[l][j] / b1t;
            double vhat_b = v_b_[l][j] / b2t;
            double dB = -lr_ * mhat_b / (std::sqrt(vhat_b) + eps_);
            n.applyStep(dW, dB);
        }
    }
}

Momentum::Momentum(double lr, double mu) : lr_(lr), mu_(mu) {}

void Momentum::step(NeuralNetwork& net, std::size_t batchSize) {
    if (batchSize == 0) batchSize = 1;
    auto& layers = net.layers();
    if (v_w_.size() != layers.size()) {
        v_w_.resize(layers.size());
        v_b_.resize(layers.size());
        for (std::size_t l = 0; l < layers.size(); ++l) {
            auto& neurons = layers[l].neurons();
            v_w_[l].resize(neurons.size());
            v_b_[l].assign(neurons.size(), 0.0);
            for (std::size_t j = 0; j < neurons.size(); ++j)
                v_w_[l][j].assign(neurons[j].weights().size(), 0.0);
        }
    }
    for (std::size_t l = 0; l < layers.size(); ++l) {
        auto& neurons = layers[l].neurons();
        for (std::size_t j = 0; j < neurons.size(); ++j) {
            auto& n = neurons[j];
            Vector dW(n.weights().size());
            for (std::size_t i = 0; i < dW.size(); ++i) {
                double g = n.gradWeights()[i] / double(batchSize);
                v_w_[l][j][i] = mu_ * v_w_[l][j][i] - lr_ * g;
                dW[i] = v_w_[l][j][i];
            }
            double gb = n.gradBias() / double(batchSize);
            v_b_[l][j] = mu_ * v_b_[l][j] - lr_ * gb;
            n.applyStep(dW, v_b_[l][j]);
        }
    }
}

std::unique_ptr<IOptimizer> OptimizerFactory::create(const std::string& name, double lr) {
    if (name == "sgd") return std::make_unique<SGD>(lr);
    if (name == "momentum") return std::make_unique<Momentum>(lr);
    if (name == "adam") return std::make_unique<Adam>(lr);
    throw std::invalid_argument("OptimizerFactory: unknown optimizer '" + name +
                                "' (choose sgd|momentum|adam)");
}

} // namespace miniann
