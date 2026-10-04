#pragma once
// Configuration validation as a Composite of small single-purpose rules.
// Each rule is independently testable; ConfigValidator::defaults() assembles
// the production set. Hard errors block training, advisories (§7 of the work
// bench spec) merely warn so experimentation stays possible.
#include "miniann/experiment.hpp"
#include <memory>
#include <string>
#include <vector>

namespace miniann {

struct ValidationReport {
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
};

class IValidationRule {
public:
    virtual ~IValidationRule() = default;
    virtual void check(const ExperimentConfig& cfg, const PreparedData& data,
                       ValidationReport& out) const = 0;
    virtual std::string name() const = 0;
};

// ---- hard errors (block training) ----
class HiddenLayersRule : public IValidationRule {
public:
    void check(const ExperimentConfig& cfg, const PreparedData&,
               ValidationReport& out) const override;
    std::string name() const override { return "hidden-layers"; }
};

class HiddenSizesRule : public IValidationRule {
public:
    void check(const ExperimentConfig& cfg, const PreparedData&,
               ValidationReport& out) const override;
    std::string name() const override { return "hidden-sizes"; }
};

class HiddenActivationsRule : public IValidationRule {
public:
    void check(const ExperimentConfig& cfg, const PreparedData&,
               ValidationReport& out) const override;
    std::string name() const override { return "hidden-activations"; }
};

class OutputActivationRule : public IValidationRule {
public:
    void check(const ExperimentConfig& cfg, const PreparedData&,
               ValidationReport& out) const override;
    std::string name() const override { return "output-activation"; }
};

class LossKnownRule : public IValidationRule {
public:
    void check(const ExperimentConfig& cfg, const PreparedData&,
               ValidationReport& out) const override;
    std::string name() const override { return "loss-known"; }
};

class OptimizerKnownRule : public IValidationRule {
public:
    void check(const ExperimentConfig& cfg, const PreparedData&,
               ValidationReport& out) const override;
    std::string name() const override { return "optimizer-known"; }
};

class EpochRangeRule : public IValidationRule {
public:
    void check(const ExperimentConfig& cfg, const PreparedData&,
               ValidationReport& out) const override;
    std::string name() const override { return "epoch-range"; }
};

class BceOutputRule : public IValidationRule {
public:
    void check(const ExperimentConfig& cfg, const PreparedData&,
               ValidationReport& out) const override;
    std::string name() const override { return "bce-output"; }
};

// ---- advisories (warn, never block) ----
class MseMultiOutputNote : public IValidationRule {
public:
    void check(const ExperimentConfig& cfg, const PreparedData&,
               ValidationReport& out) const override;
    std::string name() const override { return "mse-multi-output-note"; }
};

class LinearCceNote : public IValidationRule {
public:
    void check(const ExperimentConfig& cfg, const PreparedData&,
               ValidationReport& out) const override;
    std::string name() const override { return "linear-cce-note"; }
};

class BceNonSigmoidNote : public IValidationRule {
public:
    void check(const ExperimentConfig& cfg, const PreparedData&,
               ValidationReport& out) const override;
    std::string name() const override { return "bce-non-sigmoid-note"; }
};

class EffectiveBatchNote : public IValidationRule {
public:
    void check(const ExperimentConfig& cfg, const PreparedData&,
               ValidationReport& out) const override;
    std::string name() const override { return "effective-batch-note"; }
};

class TinyDatasetNote : public IValidationRule {
public:
    void check(const ExperimentConfig& cfg, const PreparedData&,
               ValidationReport& out) const override;
    std::string name() const override { return "tiny-dataset-note"; }
};

class ConfigValidator {
public:
    void addRule(std::unique_ptr<IValidationRule> rule);
    std::size_t ruleCount() const { return rules_.size(); }
    // Returns true when no hard errors fired.
    bool validate(const ExperimentConfig& cfg, const PreparedData& data,
                  std::string& err, std::string& warn) const;
    static ConfigValidator defaults();
private:
    std::vector<std::unique_ptr<IValidationRule>> rules_;
};

} // namespace miniann
