// Round-trip + activation sanity (sigma(0)=0.5, relu(-2)=0, dim-mismatch throws).
#include <iostream>
#include <random>
#include <cassert>
#include <cmath>
#include "miniann/network.hpp"
#include "miniann/activation.hpp"
#include "miniann/serializer.hpp"

using namespace miniann;

int main() {
    Sigmoid s; Tanh t; ReLU r;
    assert(std::abs(s.activate(0.0) - 0.5) < 1e-12);
    assert(r.activate(-2.0) == 0.0);
    assert(std::abs(t.activate(0.0)) < 1e-12);

    std::mt19937 rng(7);
    NeuralNetwork net;
    net.addLayer(Layer(2, 2, ActivationFactory::create("tanh"), rng));
    net.addLayer(Layer(1, 2, ActivationFactory::create("sigmoid"), rng));
    Vector x = {1.0, 0.0};
    Vector y0 = net.predict(x);
    ModelSerializer::save(net, "test_roundtrip.model");
    NeuralNetwork net2 = ModelSerializer::load("test_roundtrip.model", rng);
    Vector y1 = net2.predict(x);
    double d = std::abs(y0[0] - y1[0]);
    std::cout << "roundtrip diff=" << d << "\n";
    if (d > 1e-14) { std::cout << "SERIALIZER FAIL\n"; return 1; }
    bool threw = false;
    try { net.addLayer(Layer(1, 99, ActivationFactory::create("sigmoid"), rng)); }
    catch (const std::invalid_argument&) { threw = true; }
    if (!threw) { std::cout << "DIM CHECK FAIL\n"; return 1; }
    std::cout << "ALL MVP CHECKS PASS\n";
    return 0;
}
