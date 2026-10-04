#pragma once
#include <string>
#include <cmath>
#include <memory>
#include <algorithm>

namespace miniann {

/**
 * Strategy Pattern: Learning Rate Schedulers.
 * Dynamically modulates the optimizer learning rate as training progresses,
 * preventing late-stage oscillations and accelerating convergence.
 */
class ILearningRateScheduler {
public:
    virtual ~ILearningRateScheduler() = default;
    virtual double getRate(double baseLr, int epoch, int totalEpochs) const = 0;
    virtual std::string name() const = 0;
    virtual std::unique_ptr<ILearningRateScheduler> clone() const = 0;
};

class ConstantLR : public ILearningRateScheduler {
public:
    double getRate(double baseLr, int /*epoch*/, int /*totalEpochs*/) const override {
        return baseLr;
    }
    std::string name() const override { return "Constant"; }
    std::unique_ptr<ILearningRateScheduler> clone() const override {
        return std::make_unique<ConstantLR>(*this);
    }
};

class StepDecayLR : public ILearningRateScheduler {
public:
    StepDecayLR(int stepSize = 300, double gamma = 0.5)
        : stepSize_(stepSize > 0 ? stepSize : 100), gamma_(gamma > 0.0 ? gamma : 0.5) {}

    double getRate(double baseLr, int epoch, int /*totalEpochs*/) const override {
        int drops = std::max(0, epoch / stepSize_);
        return baseLr * std::pow(gamma_, drops);
    }
    std::string name() const override { return "Step Decay"; }
    std::unique_ptr<ILearningRateScheduler> clone() const override {
        return std::make_unique<StepDecayLR>(*this);
    }

private:
    int stepSize_;
    double gamma_;
};

class CosineAnnealingLR : public ILearningRateScheduler {
public:
    CosineAnnealingLR(double minLr = 0.0001) : minLr_(minLr) {}

    double getRate(double baseLr, int epoch, int totalEpochs) const override {
        if (totalEpochs <= 1) return baseLr;
        double progress = std::clamp(double(epoch) / double(totalEpochs), 0.0, 1.0);
        return minLr_ + 0.5 * (baseLr - minLr_) * (1.0 + std::cos(progress * 3.14159265358979323846));
    }
    std::string name() const override { return "Cosine Annealing"; }
    std::unique_ptr<ILearningRateScheduler> clone() const override {
        return std::make_unique<CosineAnnealingLR>(*this);
    }

private:
    double minLr_;
};

class SchedulerFactory {
public:
    static std::unique_ptr<ILearningRateScheduler> create(const std::string& name,
                                                          int stepSize = 300,
                                                          double gamma = 0.5,
                                                          double minLr = 0.0001) {
        if (name == "step" || name == "Step" || name == "Step Decay") {
            return std::make_unique<StepDecayLR>(stepSize, gamma);
        }
        if (name == "cosine" || name == "Cosine" || name == "Cosine Annealing") {
            return std::make_unique<CosineAnnealingLR>(minLr);
        }
        return std::make_unique<ConstantLR>();
    }
};

} // namespace miniann
