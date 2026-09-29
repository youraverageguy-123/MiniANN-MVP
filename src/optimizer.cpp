#include "miniann/optimizer.hpp"
#include <cmath>

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

} // namespace miniann
