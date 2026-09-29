// Gradient check: centered finite differences vs analytical backprop.
// Relative error must be < 1e-6 (spec section 10.1).
#include <iostream>
#include <random>
#include <cmath>
#include "miniann/network.hpp"
#include "miniann/loss.hpp"

using namespace miniann;

int main() {
    std::mt19937 rng(123);
    NeuralNetwork net;
    net.addLayer(Layer(3, 2, ActivationFactory::create("tanh"), rng));
    net.addLayer(Layer(1, 3, ActivationFactory::create("sigmoid"), rng));
    MSELoss loss;
    Vector x = {0.5, -0.3};
    Vector y = {1.0};
    const double eps = 1e-5;

    // analytical grads
    net.zeroGradients();
    Vector pred = net.predict(x);
    net.backward(loss.gradient(pred, y));
    std::vector<double> analytic;
    for (auto& l : net.layers())
        for (auto& n : l.neurons()) {
            for (double g : n.gradWeights()) analytic.push_back(g);
            analytic.push_back(n.gradBias());
        }

    // numeric grads
    std::vector<double> numeric;
    auto lossAt = [&](NeuralNetwork& n) {
        return loss.compute(n.predict(x), y);
    };
    for (auto& l : net.layers()) {
        for (auto& n : l.neurons()) {
            for (std::size_t i = 0; i < n.weights().size(); ++i) {
                Vector w = n.weights(); double b = n.bias();
                Vector wp = w; wp[i] += eps;
                n.setParameters(wp, b); double lp = lossAt(net);
                Vector wm = w; wm[i] -= eps;
                n.setParameters(wm, b); double lm = lossAt(net);
                n.setParameters(w, b);
                numeric.push_back((lp - lm) / (2 * eps));
            }
            {
                Vector w = n.weights(); double b = n.bias();
                n.setParameters(w, b + eps); double lp = lossAt(net);
                n.setParameters(w, b - eps); double lm = lossAt(net);
                n.setParameters(w, b);
                numeric.push_back((lp - lm) / (2 * eps));
            }
        }
    }
    double num2 = 0, den2 = 0, diff2 = 0;
    for (std::size_t i = 0; i < analytic.size(); ++i) {
        diff2 += (numeric[i] - analytic[i]) * (numeric[i] - analytic[i]);
        num2 += numeric[i] * numeric[i];
        den2 += analytic[i] * analytic[i];
    }
    double rel = std::sqrt(diff2) / (std::max(std::sqrt(num2), std::sqrt(den2)) + 1e-8);
    std::cout << "params=" << analytic.size() << " rel_error=" << rel << "\n";
    if (rel < 1e-6) { std::cout << "GRADIENT CHECK PASS\n"; return 0; }
    std::cout << "GRADIENT CHECK FAIL\n";
    return 1;
}
