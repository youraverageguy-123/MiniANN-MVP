#pragma once
// Dataset sourcing as polymorphic objects: each source knows how to load one
// kind of dataset, and CSV target-column selection is a Chain of
// Responsibility (header-name link -> distinct-value link -> last-column
// fallback). Adding a source never touches existing code.
#include "miniann/experiment.hpp"
#include <memory>
#include <string>
#include <vector>

namespace miniann {

// Per-column profile gathered by a single pass over the file.
struct ColumnProfile {
    bool numeric = true;
    bool integral = true;
    bool capped = false; // >33 distinct values: not a label candidate
    std::vector<std::string> distinct;
};

struct TargetScan {
    int cols = 0;
    std::vector<std::string> headerNames; // lowered, unquoted (empty when no header)
    std::vector<ColumnProfile> columns;
};

struct TargetChoice {
    int column = -1;
    std::string reason;
};

class ITargetSelector {
public:
    virtual ~ITargetSelector() = default;
    // Returns true when this link claims the decision.
    virtual bool tryPick(const TargetScan& scan, TargetChoice& out) const = 0;
    virtual std::string name() const = 0;
};

class HeaderNameSelector : public ITargetSelector {
public:
    bool tryPick(const TargetScan& scan, TargetChoice& out) const override;
    std::string name() const override { return "header-name"; }
};

class DistinctValueSelector : public ITargetSelector {
public:
    bool tryPick(const TargetScan& scan, TargetChoice& out) const override;
    std::string name() const override { return "distinct-value"; }
};

class LastColumnSelector : public ITargetSelector {
public:
    bool tryPick(const TargetScan& scan, TargetChoice& out) const override;
    std::string name() const override { return "last-column"; }
};

class TargetSelectorChain {
public:
    void addLink(std::unique_ptr<ITargetSelector> link);
    std::size_t linkCount() const { return links_.size(); }
    TargetChoice select(const TargetScan& scan) const; // first claim wins
    static TargetSelectorChain defaults();
private:
    std::vector<std::unique_ptr<ITargetSelector>> links_;
};

struct LoadedRaw {
    Dataset data;
    int targetCol = 0;
    std::string note; // target provenance for the UI
    std::string name;
};

class IDatasetSource {
public:
    virtual ~IDatasetSource() = default;
    virtual std::string key() const = 0;
    virtual LoadedRaw load(const ExperimentConfig& cfg) const = 0;
};

class AndGateSource : public IDatasetSource {
public:
    std::string key() const override { return "and"; }
    LoadedRaw load(const ExperimentConfig&) const override;
};

class OrGateSource : public IDatasetSource {
public:
    std::string key() const override { return "or"; }
    LoadedRaw load(const ExperimentConfig&) const override;
};

class XorGateSource : public IDatasetSource {
public:
    std::string key() const override { return "xor"; }
    LoadedRaw load(const ExperimentConfig&) const override;
};

class IrisSource : public IDatasetSource {
public:
    std::string key() const override { return "iris"; }
    LoadedRaw load(const ExperimentConfig&) const override;
};

class CsvSource : public IDatasetSource {
public:
    std::string key() const override { return "csv"; }
    LoadedRaw load(const ExperimentConfig& cfg) const override;
};

class DatasetSourceFactory {
public:
    // "and"|"or"|"xor"|"iris"|"csv". Throws invalid_argument otherwise.
    static std::unique_ptr<IDatasetSource> create(const std::string& key);
};

} // namespace miniann
