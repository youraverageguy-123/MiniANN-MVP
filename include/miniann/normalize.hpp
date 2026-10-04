#pragma once
// Feature normalization as Strategy objects: statistics are fitted on the
// TRAIN split only, then the same transformation applies to every split.
// INormalizer::normalize() is the Template Method (fixed skeleton),
// fit()/apply() are the primitive operations subclasses provide.
#include "miniann/dataset.hpp"
#include <memory>
#include <string>

namespace miniann {

class INormalizer {
public:
    virtual ~INormalizer() = default;
    virtual void fit(const Dataset& train) = 0;
    virtual Vector apply(const Vector& x) const = 0;
    virtual std::string name() const = 0;
    Dataset normalize(const Dataset& d) const;
};

class NoNormalization : public INormalizer {
public:
    void fit(const Dataset&) override {}
    Vector apply(const Vector& x) const override { return x; }
    std::string name() const override { return "none"; }
};

class MaxAbsNormalization : public INormalizer {
public:
    // Per-feature max-abs: each feature is scaled by its own peak so
    // small-magnitude features keep a usable signal instead of being
    // squashed by the global maximum. Fit on TRAIN inputs only.
    void fit(const Dataset& train) override;
    Vector apply(const Vector& x) const override;
    std::string name() const override { return "maxabs"; }
private:
    Vector scale_;
};

class MinMaxNormalization : public INormalizer {
public:
    void fit(const Dataset& train) override;
    Vector apply(const Vector& x) const override;
    std::string name() const override { return "minmax"; }
private:
    Vector lo_, hi_;
};

class NormalizerFactory {
public:
    // 0 = none, 1 = max-abs, 2 = min-max. Throws invalid_argument otherwise.
    static std::unique_ptr<INormalizer> create(int mode);
    static std::unique_ptr<INormalizer> create(const std::string& name);
};

} // namespace miniann
