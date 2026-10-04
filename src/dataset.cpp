#include "miniann/dataset.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace miniann {

void Dataset::add(Vector input, Vector target) {
    if (!inputs_.empty()) {
        if (input.size() != inputs_[0].size())
            throw std::invalid_argument("Dataset::add: inconsistent input dim");
        if (target.size() != targets_[0].size())
            throw std::invalid_argument("Dataset::add: inconsistent target dim");
    }
    inputs_.push_back(std::move(input));
    targets_.push_back(std::move(target));
}

void Dataset::shuffle(std::mt19937& rng) {
    std::vector<std::size_t> idx(inputs_.size());
    for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng);
    std::vector<Vector> ni, nt;
    ni.reserve(inputs_.size()); nt.reserve(targets_.size());
    for (auto k : idx) { ni.push_back(inputs_[k]); nt.push_back(targets_[k]); }
    inputs_ = std::move(ni); targets_ = std::move(nt);
}

std::pair<Dataset, Dataset> Dataset::split(double trainFraction, std::mt19937& rng) const {
    if (inputs_.empty()) throw std::runtime_error("Dataset::split: empty dataset");
    if (trainFraction <= 0.0 || trainFraction >= 1.0)
        throw std::invalid_argument("Dataset::split: trainFraction must be in (0,1)");
    std::vector<std::size_t> idx(inputs_.size());
    for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng);
    std::size_t nTrain = std::size_t(double(inputs_.size()) * trainFraction);
    Dataset tr, te;
    for (std::size_t k = 0; k < idx.size(); ++k) {
        if (k < nTrain) tr.add(inputs_[idx[k]], targets_[idx[k]]);
        else te.add(inputs_[idx[k]], targets_[idx[k]]);
    }
    return {tr, te};
}

std::tuple<Dataset, Dataset, Dataset>
Dataset::splitTrainValTest(double trainFraction, double valFraction, unsigned seed) const {
    if (inputs_.empty()) throw std::runtime_error("Dataset::splitTrainValTest: empty dataset");
    if (trainFraction <= 0.0 || trainFraction >= 1.0 || valFraction < 0.0 || valFraction >= 1.0 ||
        trainFraction + valFraction >= 1.0)
        throw std::invalid_argument("Dataset::splitTrainValTest: need train>0, val>=0, train+val<1");
    std::vector<std::size_t> idx(inputs_.size());
    for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);
    std::size_t nTrain = std::size_t(double(inputs_.size()) * trainFraction);
    std::size_t nVal = std::size_t(double(inputs_.size()) * valFraction);
    if (nTrain == 0 || nVal == 0 || nTrain + nVal >= inputs_.size())
        throw std::invalid_argument("Dataset::splitTrainValTest: split leaves an empty part");
    Dataset tr, va, te;
    for (std::size_t k = 0; k < idx.size(); ++k) {
        if (k < nTrain) tr.add(inputs_[idx[k]], targets_[idx[k]]);
        else if (k < nTrain + nVal) va.add(inputs_[idx[k]], targets_[idx[k]]);
        else te.add(inputs_[idx[k]], targets_[idx[k]]);
    }
    return {tr, va, te};
}

void Dataset::ShuffledOrder::order(std::vector<std::size_t>& idx, unsigned seed) const {
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);
}

std::tuple<Dataset, Dataset, Dataset>
Dataset::splitCounts(std::size_t nTrain, std::size_t nVal, const ISplitOrder& order, unsigned seed) const {
    if (inputs_.empty()) throw std::runtime_error("Dataset::splitCounts: empty dataset");
    if (nTrain == 0) throw std::invalid_argument("Dataset::splitCounts: need >= 1 train sample");
    if (nTrain + nVal >= inputs_.size())
        throw std::invalid_argument("Dataset::splitCounts: need >= 1 test sample");
    std::vector<std::size_t> idx(inputs_.size());
    for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    order.order(idx, seed);
    Dataset tr, va, te;
    for (std::size_t k = 0; k < idx.size(); ++k) {
        if (k < nTrain) tr.add(inputs_[idx[k]], targets_[idx[k]]);
        else if (k < nTrain + nVal) va.add(inputs_[idx[k]], targets_[idx[k]]);
        else te.add(inputs_[idx[k]], targets_[idx[k]]);
    }
    return {tr, va, te};
}

std::tuple<Dataset, Dataset, Dataset>
Dataset::splitCounts(std::size_t nTrain, std::size_t nVal, unsigned seed, bool doShuffle) const {
    if (doShuffle) return splitCounts(nTrain, nVal, ShuffledOrder(), seed);
    return splitCounts(nTrain, nVal, SequentialOrder(), seed);
}

void Dataset::validate() const {
    if (inputs_.empty()) throw std::runtime_error("Dataset::validate: empty dataset");
    std::size_t ni = inputs_[0].size(), nt = targets_[0].size();
    if (ni == 0 || nt == 0) throw std::runtime_error("Dataset::validate: zero dimension");
    for (std::size_t i = 0; i < inputs_.size(); ++i) {
        if (inputs_[i].size() != ni || targets_[i].size() != nt)
            throw std::runtime_error("Dataset::validate: dimension mismatch");
        for (double v : inputs_[i])
            if (!std::isfinite(v)) throw std::runtime_error("Dataset::validate: NaN/Inf in inputs");
        for (double v : targets_[i])
            if (!std::isfinite(v)) throw std::runtime_error("Dataset::validate: NaN/Inf in targets");
    }
}

// RFC-4180-ish: handles quoted fields, trims whitespace, skips blank lines.
static std::vector<std::string> parseLine(const std::string& line) {
    std::vector<std::string> cells;
    std::string cur;
    bool inQ = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (inQ) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') { cur += '"'; ++i; }
                else inQ = false;
            } else cur += c;
        } else {
            if (c == '"') inQ = true;
            else if (c == ',') { cells.push_back(cur); cur.clear(); }
            else cur += c;
        }
    }
    cells.push_back(cur);
    for (auto& s : cells) {
        std::size_t a = s.find_first_not_of(" \t\r\n");
        std::size_t b = s.find_last_not_of(" \t\r\n");
        s = (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
    }
    return cells;
}

Dataset Dataset::loadCSV(const std::string& path, bool hasHeader,
                         int targetColIndex, int targetDim) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Dataset::loadCSV: cannot open " + path);
    if (targetDim <= 0) throw std::invalid_argument("Dataset::loadCSV: targetDim must be > 0");
    if (targetColIndex < -1) throw std::invalid_argument("Dataset::loadCSV: targetColIndex must be >= -1");
    Dataset ds;
    std::string line;
    bool first = true;
    std::size_t nCols = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (first && hasHeader) { first = false; continue; }
        first = false;
        auto cells = parseLine(line);
        if (cells.size() == 1 && cells[0].empty()) continue;
        if (nCols == 0) nCols = cells.size();
        if (cells.size() != nCols)
            throw std::runtime_error("Dataset::loadCSV: ragged row (inconsistent column count)");
        // If targetColIndex == -1, treat it as "last column(s)".
        int tc = targetColIndex;
        if (tc == -1) tc = int(nCols) - targetDim;
        if (tc < 0 || tc + targetDim > int(cells.size()))
            throw std::runtime_error("Dataset::loadCSV: target range out of bounds");
        std::vector<double> row;
        row.reserve(cells.size());
        for (auto& c : cells) {
            try { row.push_back(std::stod(c)); }
            catch (...) { throw std::runtime_error("Dataset::loadCSV: non-numeric value '" + c + "'"); }
        }
        Vector tgt(row.begin() + tc, row.begin() + tc + targetDim);
        Vector inp;
        for (int i = 0; i < int(row.size()); ++i)
            if (i < tc || i >= tc + targetDim) inp.push_back(row[i]);
        ds.add(std::move(inp), std::move(tgt));
    }
    if (ds.size() == 0) throw std::runtime_error("Dataset::loadCSV: no data rows");
    return ds;
}

Dataset makeAndGate() {
    Dataset d;
    d.add({0,0},{0}); d.add({0,1},{0}); d.add({1,0},{0}); d.add({1,1},{1});
    return d;
}
Dataset makeOrGate() {
    Dataset d;
    d.add({0,0},{0}); d.add({0,1},{1}); d.add({1,0},{1}); d.add({1,1},{1});
    return d;
}
Dataset makeXorGate() {
    Dataset d;
    d.add({0,0},{0}); d.add({0,1},{1}); d.add({1,0},{1}); d.add({1,1},{0});
    return d;
}

} // namespace miniann
