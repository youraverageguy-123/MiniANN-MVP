#pragma once
#include "miniann/types.hpp"
#include <string>
#include <random>
#include <utility>
#include <tuple>

namespace miniann {

class Dataset {
public:
    // Strategy for index ordering before a split: seeded shuffle for
    // reproducibility, or sequential (first rows train) when the user
    // disables split shuffling. Replaces a boolean flag with polymorphism.
    class ISplitOrder {
    public:
        virtual ~ISplitOrder() = default;
        virtual void order(std::vector<std::size_t>& idx, unsigned seed) const = 0;
    };
    class ShuffledOrder : public ISplitOrder {
    public:
        void order(std::vector<std::size_t>& idx, unsigned seed) const override;
    };
    class SequentialOrder : public ISplitOrder {
    public:
        void order(std::vector<std::size_t>& idx, unsigned seed) const override;
    };
    void add(Vector input, Vector target);
    std::size_t size() const { return inputs_.size(); }
    const Vector& input(std::size_t i) const { return inputs_.at(i); }
    const Vector& target(std::size_t i) const { return targets_.at(i); }
    void shuffle(std::mt19937& rng);
    std::pair<Dataset, Dataset> split(double trainFraction, std::mt19937& rng) const;
    // Deterministic (train, validation, test) split: shuffles a copy of the
    // index order with mt19937(seed), so the same seed always yields the same
    // split. Throws on empty data or fractions that leave any part empty.
    std::tuple<Dataset, Dataset, Dataset>
    splitTrainValTest(double trainFraction, double valFraction, unsigned seed) const;
    // Exact-count variant used when minimum part sizes are enforced
    // (each part gets >= 1 sample when the dataset allows it). When
    // doShuffle is false the split is sequential (first rows train), which is
    // still fully reproducible. Throws if counts don't fit the dataset.
    std::tuple<Dataset, Dataset, Dataset>
    splitCounts(std::size_t nTrain, std::size_t nVal, const ISplitOrder& order, unsigned seed) const;
    // Convenience overload preserving the old boolean form.
    std::tuple<Dataset, Dataset, Dataset>
    splitCounts(std::size_t nTrain, std::size_t nVal, unsigned seed, bool doShuffle = true) const;
    void validate() const; // throws on empty / dim mismatch / NaN-Inf
    static Dataset loadCSV(const std::string& path, bool hasHeader,
                           int targetColIndex, int targetDim);
private:
    std::vector<Vector> inputs_;
    std::vector<Vector> targets_;
};

Dataset makeAndGate();
Dataset makeOrGate();
Dataset makeXorGate();

} // namespace miniann
