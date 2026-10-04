#include "miniann/validate.hpp"
#include <sstream>

namespace miniann {

void HiddenLayersRule::check(const ExperimentConfig& cfg, const PreparedData&,
                             ValidationReport& out) const {
    if (cfg.hidden.empty() || cfg.hidden.size() != cfg.hiddenActs.size())
        out.errors.push_back("Need at least 1 hidden layer.");
}

void HiddenSizesRule::check(const ExperimentConfig& cfg, const PreparedData&,
                            ValidationReport& out) const {
    for (std::size_t i = 0; i < cfg.hidden.size(); ++i) {
        if (cfg.hidden[i] < 1 || cfg.hidden[i] > 1024) {
            out.errors.push_back("Hidden layer " + std::to_string(i + 1) +
                                 ": neuron count must be 1..1024.");
            return;
        }
    }
}

void HiddenActivationsRule::check(const ExperimentConfig& cfg, const PreparedData&,
                                  ValidationReport& out) const {
    for (auto& a : cfg.hiddenActs) {
        try { ActivationFactory::create(a); }
        catch (const std::exception&) {
            out.errors.push_back("Unknown hidden activation '" + a + "'.");
            return;
        }
    }
}

void OutputActivationRule::check(const ExperimentConfig& cfg, const PreparedData&,
                                 ValidationReport& out) const {
    try { ActivationFactory::create(cfg.outputAct); }
    catch (const std::exception&) {
        out.errors.push_back("Unknown output activation '" + cfg.outputAct + "'.");
    }
}

void LossKnownRule::check(const ExperimentConfig& cfg, const PreparedData&,
                          ValidationReport& out) const {
    try { LossFactory::create(cfg.loss); }
    catch (const std::exception&) {
        out.errors.push_back("Unknown loss '" + cfg.loss + "'.");
    }
}

void OptimizerKnownRule::check(const ExperimentConfig& cfg, const PreparedData&,
                               ValidationReport& out) const {
    try { OptimizerFactory::create(cfg.optimizer, cfg.opt); }
    catch (const std::exception& e) { out.errors.push_back(e.what()); }
}

void EpochRangeRule::check(const ExperimentConfig& cfg, const PreparedData&,
                           ValidationReport& out) const {
    if (cfg.epochs < 1 || cfg.epochs > 200000)
        out.errors.push_back("Epoch count must be 1..200000.");
}

void BceOutputRule::check(const ExperimentConfig& cfg, const PreparedData& data,
                          ValidationReport& out) const {
    if (cfg.loss == "bce" && data.outDim != 1)
        out.errors.push_back("BCE needs a 1-output network (use MSE/CCE).");
}

void MseMultiOutputNote::check(const ExperimentConfig& cfg, const PreparedData& data,
                               ValidationReport& out) const {
    if (cfg.loss == "mse" && data.outDim > 1) {
        std::ostringstream w;
        w << "Warning: MSE on a " << data.outDim
          << "-output task — CCE usually trains classifiers better.";
        out.warnings.push_back(w.str());
    }
}

void LinearCceNote::check(const ExperimentConfig& cfg, const PreparedData&,
                          ValidationReport& out) const {
    if (cfg.loss == "cce" && cfg.outputAct == "linear")
        out.warnings.push_back("Warning: Linear outputs are unbounded — CCE expects [0,1]; prefer Sigmoid output.");
}

void BceNonSigmoidNote::check(const ExperimentConfig& cfg, const PreparedData&,
                              ValidationReport& out) const {
    if (cfg.loss == "bce" && cfg.outputAct != "sigmoid")
        out.warnings.push_back("Warning: BCE is designed for Sigmoid outputs.");
}

void EffectiveBatchNote::check(const ExperimentConfig& cfg, const PreparedData& data,
                               ValidationReport& out) const {
    std::size_t eff = (cfg.batchSize == 0 || cfg.batchSize > data.nTrain) ? data.nTrain : cfg.batchSize;
    if (cfg.batchSize > data.nTrain) {
        std::ostringstream w;
        w << "Note: batch " << cfg.batchSize << " > train size " << data.nTrain
          << " — effective batch " << eff << " (full-batch).";
        out.warnings.push_back(w.str());
    }
}

void TinyDatasetNote::check(const ExperimentConfig&, const PreparedData& data,
                            ValidationReport& out) const {
    if (data.tiny) {
        std::ostringstream w;
        w << "Note: tiny dataset — training on all " << data.samples
          << " samples; no held-out test set exists, so no test metric is reported.";
        out.warnings.push_back(w.str());
    }
}

void ConfigValidator::addRule(std::unique_ptr<IValidationRule> rule) {
    rules_.push_back(std::move(rule));
}

static std::string joinLines(const std::vector<std::string>& v) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) s += "\n";
        s += v[i];
    }
    return s;
}

bool ConfigValidator::validate(const ExperimentConfig& cfg, const PreparedData& data,
                               std::string& err, std::string& warn) const {
    ValidationReport rep;
    for (auto& r : rules_) r->check(cfg, data, rep);
    err = joinLines(rep.errors);
    warn = joinLines(rep.warnings);
    return rep.errors.empty();
}

ConfigValidator ConfigValidator::defaults() {
    ConfigValidator v;
    v.addRule(std::make_unique<HiddenLayersRule>());
    v.addRule(std::make_unique<HiddenSizesRule>());
    v.addRule(std::make_unique<HiddenActivationsRule>());
    v.addRule(std::make_unique<OutputActivationRule>());
    v.addRule(std::make_unique<LossKnownRule>());
    v.addRule(std::make_unique<OptimizerKnownRule>());
    v.addRule(std::make_unique<EpochRangeRule>());
    v.addRule(std::make_unique<BceOutputRule>());
    v.addRule(std::make_unique<MseMultiOutputNote>());
    v.addRule(std::make_unique<LinearCceNote>());
    v.addRule(std::make_unique<BceNonSigmoidNote>());
    v.addRule(std::make_unique<EffectiveBatchNote>());
    return v;
}

} // namespace miniann
