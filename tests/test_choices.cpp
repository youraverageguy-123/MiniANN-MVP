// Checks LeakyReLU/Swish values + analytic derivatives vs centered
// finite differences, plus factory sanity (activations + optimizers).
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include "miniann/activation.hpp"
#include "miniann/loss.hpp"
#include "miniann/optimizer.hpp"

using namespace miniann;

static double numDeriv(const IActivation& f, double z) {
    const double e = 1e-6;
    return (f.activate(z + e) - f.activate(z - e)) / (2 * e);
}

int main() {
    LeakyReLU l(0.01);
    assert(l.activate(2.0) == 2.0);
    assert(std::abs(l.activate(-4.0) + 0.04) < 1e-12);
    assert(l.derivative(5.0) == 1.0 && l.derivative(-5.0) == 0.01);
    assert(std::abs(l.name() == "leaky_relu" ? 0 : 1) == 0);

    Swish s;
    assert(std::abs(s.activate(0.0)) < 1e-12);
    assert(std::abs(s.derivative(0.0) - 0.5) < 1e-12); // sigmoid(0)=0.5
    for (double z : {-2.0, -0.5, 0.7, 3.0}) {
        double d = std::abs(s.derivative(z) - numDeriv(s, z));
        double denom = std::max(std::abs(s.derivative(z)), 1e-8);
        assert(d / denom < 1e-6);
        double dl = std::abs(l.derivative(z) - numDeriv(l, z));
        assert(dl < 1e-6 || z == 0.0); // kink only exactly at 0
    }

    // factories resolve every advertised token, reject garbage
    for (const char* t : {"sigmoid", "tanh", "relu", "leaky_relu", "swish", "linear", "softmax"})
        assert(ActivationFactory::create(t) != nullptr);
    for (const char* t : {"sgd", "momentum", "adam"})
        assert(OptimizerFactory::create(t, 0.01) != nullptr);
    for (const char* t : {"mse", "bce", "cce"})
        assert(LossFactory::create(t) != nullptr);
    bool threw = false;
    try { ActivationFactory::create("soft_max_typo"); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    threw = false;
    try { OptimizerFactory::create("newton", 0.01); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    threw = false;
    try { LossFactory::create("hinge"); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);

    // BCE/CCE values + analytic gradients vs centered finite differences
    {
        BCELoss bce;
        assert(std::abs(bce.compute({0.5}, {1.0}) - 0.6931471805599453) < 1e-9);
        Vector g = bce.gradient({0.7}, {1.0});
        double num = (bce.compute({0.7 + 1e-6}, {1.0}) - bce.compute({0.7 - 1e-6}, {1.0})) / 2e-6;
        assert(std::abs(g[0] - num) / std::max(std::abs(num), 1e-8) < 1e-6);
        bool bad = false;
        try { bce.compute({0.5, 0.5}, {1.0, 0.0}); } catch (const std::invalid_argument&) { bad = true; }
        assert(bad); // binary loss rejects multi-output
    }
    {
        CCELoss cce;
        Vector p = {0.7, 0.2, 0.1}, t = {1.0, 0.0, 0.0};
        Vector g = cce.gradient(p, t);
        for (std::size_t i = 0; i < p.size(); ++i) {
            Vector pp = p;
            pp[i] += 1e-6;
            Vector pm = p;
            pm[i] -= 1e-6;
            double num = (cce.compute(pp, t) - cce.compute(pm, t)) / 2e-6;
            assert(std::abs(g[i] - num) / std::max(std::abs(num), 1e-8) < 1e-6);
        }
        // wrong classes must be pushed down (positive gradient), not ignored
        assert(g[1] > 0.0 && g[2] > 0.0);
        assert(g[0] < 0.0); // correct class pushed up
        // symmetric case: -(log .5 + log .5)/2
        assert(std::abs(cce.compute({0.5, 0.5}, {1.0, 0.0}) - 0.6931471805599453) < 1e-9);
    }

    std::cout << "NEW ACTIVATIONS + FACTORIES PASS\n";
    return 0;
}
