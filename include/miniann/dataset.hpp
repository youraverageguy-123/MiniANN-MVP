#pragma once
#include "miniann/types.hpp"
#include <string>
#include <random>
#include <utility>

namespace miniann {

class Dataset {
public:
    void add(Vector input, Vector target);
    std::size_t size() const { return inputs_.size(); }
    const Vector& input(std::size_t i) const { return inputs_.at(i); }
    const Vector& target(std::size_t i) const { return targets_.at(i); }
    void shuffle(std::mt19937& rng);
    std::pair<Dataset, Dataset> split(double trainFraction, std::mt19937& rng) const;
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
