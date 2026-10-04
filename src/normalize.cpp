#include "miniann/normalize.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace miniann {

Dataset INormalizer::normalize(const Dataset& d) const {
    Dataset out;
    for (std::size_t i = 0; i < d.size(); ++i)
        out.add(apply(d.input(i)), d.target(i));
    return out;
}

void MaxAbsNormalization::fit(const Dataset& train) {
    scale_.clear();
    if (train.size() == 0) return;
    scale_.assign(train.input(0).size(), 0.0);
    for (std::size_t i = 0; i < train.size(); ++i) {
        const Vector& x = train.input(i);
        for (std::size_t j = 0; j < x.size() && j < scale_.size(); ++j)
            scale_[j] = std::max(scale_[j], std::abs(x[j]));
    }
    for (auto& s : scale_)
        if (s == 0.0) s = 1.0;
}

Vector MaxAbsNormalization::apply(const Vector& x) const {
    Vector y = x;
    for (std::size_t j = 0; j < y.size() && j < scale_.size(); ++j) y[j] /= scale_[j];
    return y;
}

void MinMaxNormalization::fit(const Dataset& train) {
    if (train.size() == 0) return;
    lo_ = train.input(0);
    hi_ = train.input(0);
    for (std::size_t i = 1; i < train.size(); ++i) {
        const Vector& x = train.input(i);
        for (std::size_t j = 0; j < x.size(); ++j) {
            lo_[j] = std::min(lo_[j], x[j]);
            hi_[j] = std::max(hi_[j], x[j]);
        }
    }
}

Vector MinMaxNormalization::apply(const Vector& x) const {
    Vector y = x;
    for (std::size_t j = 0; j < y.size() && j < lo_.size(); ++j) {
        double r = hi_[j] - lo_[j];
        y[j] = (r == 0.0) ? 0.0 : (y[j] - lo_[j]) / r;
    }
    return y;
}

std::unique_ptr<INormalizer> NormalizerFactory::create(int mode) {
    switch (mode) {
        case 0: return std::make_unique<NoNormalization>();
        case 1: return std::make_unique<MaxAbsNormalization>();
        case 2: return std::make_unique<MinMaxNormalization>();
    }
    throw std::invalid_argument("NormalizerFactory: unknown mode (use 0=none, 1=maxabs, 2=minmax)");
}

std::unique_ptr<INormalizer> NormalizerFactory::create(const std::string& name) {
    if (name == "none") return std::make_unique<NoNormalization>();
    if (name == "maxabs") return std::make_unique<MaxAbsNormalization>();
    if (name == "minmax") return std::make_unique<MinMaxNormalization>();
    throw std::invalid_argument("NormalizerFactory: unknown normalization '" + name +
                                "' (choose none|maxabs|minmax)");
}

} // namespace miniann
