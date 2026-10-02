#include "miniann/activation.hpp"

namespace miniann {

ActivationPtr ActivationFactory::create(const std::string& name) {
    if (name == "sigmoid") return std::make_shared<Sigmoid>();
    if (name == "tanh") return std::make_shared<Tanh>();
    if (name == "relu") return std::make_shared<ReLU>();
    if (name == "leaky_relu" || name == "leakyrelu") return std::make_shared<LeakyReLU>();
    if (name == "swish") return std::make_shared<Swish>();
    throw std::invalid_argument("ActivationFactory: unknown activation '" + name + "'");
}

} // namespace miniann
