#include "miniann/datasource.hpp"
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace miniann {

namespace {

std::string trimCell(std::string s) {
    std::size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    std::size_t b = s.find_last_not_of(" \t\r\n");
    s = s.substr(a, b - a + 1);
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
        s = s.substr(1, s.size() - 2);
    return s;
}

bool looksIntegral(const std::string& s) {
    if (s.empty()) return false;
    std::size_t i = (s[0] == '-' || s[0] == '+') ? 1 : 0;
    bool digit = false;
    for (; i < s.size(); ++i) {
        if (s[i] >= '0' && s[i] <= '9') { digit = true; continue; }
        for (std::size_t j = i; j < s.size(); ++j)
            if (s[j] != '.' && s[j] != '0') return false;
        break;
    }
    return digit;
}

bool isLabelName(const std::string& n) {
    return n == "label" || n == "target" || n == "class" || n == "y" ||
           n == "output" || n == "digit" || n == "category";
}

std::mutex g_scanMutex;
struct ScanCacheEntry {
    long long size = -1;
    std::filesystem::file_time_type mtime{};
    TargetScan scan;
};
std::map<std::string, ScanCacheEntry> g_scanCache;

} // namespace

TargetScan scanColumns(const std::string& path, bool header);
int csvColumnCount(const std::string& path);

bool HeaderNameSelector::tryPick(const TargetScan& scan, TargetChoice& out) const {
    for (std::size_t i = 0; i < scan.headerNames.size() && i < scan.columns.size(); ++i) {
        if (isLabelName(scan.headerNames[i]) && scan.columns[i].numeric) {
            out.column = (int)i;
            out.reason = "header name '" + scan.headerNames[i] + "'";
            return true;
        }
    }
    return false;
}

bool DistinctValueSelector::tryPick(const TargetScan& scan, TargetChoice& out) const {
    for (std::size_t i = 0; i < scan.columns.size(); ++i) {
        const ColumnProfile& p = scan.columns[i];
        if (p.numeric && p.integral && !p.capped && p.distinct.size() >= 2) {
            out.column = (int)i;
            out.reason = std::to_string(p.distinct.size()) + "-class label column";
            return true;
        }
    }
    return false;
}

bool LastColumnSelector::tryPick(const TargetScan& scan, TargetChoice& out) const {
    if (scan.cols <= 0) return false;
    out.column = scan.cols - 1;
    out.reason = "no label-like column found, using last";
    return true;
}

void TargetSelectorChain::addLink(std::unique_ptr<ITargetSelector> link) {
    links_.push_back(std::move(link));
}

TargetChoice TargetSelectorChain::select(const TargetScan& scan) const {
    TargetChoice out;
    for (auto& link : links_)
        if (link->tryPick(scan, out)) return out;
    return out; // column stays -1: no link claimed (empty scan)
}

TargetSelectorChain TargetSelectorChain::defaults() {
    TargetSelectorChain chain;
    chain.addLink(std::make_unique<HeaderNameSelector>());
    chain.addLink(std::make_unique<DistinctValueSelector>());
    chain.addLink(std::make_unique<LastColumnSelector>());
    return chain;
}

TargetScan scanColumns(const std::string& path, bool header) {
    {
        std::lock_guard<std::mutex> lock(g_scanMutex);
        try {
            auto it = g_scanCache.find(path);
            if (it != g_scanCache.end() &&
                it->second.size == (long long)std::filesystem::file_size(path) &&
                it->second.mtime == std::filesystem::last_write_time(path))
                return it->second.scan;
        } catch (const std::exception&) { /* stat failed: rescan below */ }
    }
    TargetScan out;
    std::ifstream file(path);
    if (!file.is_open()) return out;
    std::string line;
    bool first = true;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::vector<std::string> cells;
        {
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ',')) cells.push_back(trimCell(cell));
        }
        if (first) {
            first = false;
            out.cols = (int)cells.size();
            out.columns.assign(out.cols, ColumnProfile());
            if (header) {
                for (auto& c : cells) {
                    std::string n = c;
                    for (auto& ch : n) ch = (char)std::tolower((unsigned char)ch);
                    out.headerNames.push_back(n);
                }
                continue;
            }
        }
        if ((int)cells.size() < out.cols) continue; // ragged: loadCSV reports it
        for (int i = 0; i < out.cols; ++i) {
            ColumnProfile& p = out.columns[(std::size_t)i];
            const std::string& v = cells[(std::size_t)i];
            if (!looksIntegral(v)) {
                try { (void)std::stod(v); }
                catch (const std::exception&) { p.numeric = false; continue; }
                p.integral = false;
            }
            if (p.capped) continue;
            bool seen = false;
            for (auto& d : p.distinct)
                if (d == v) { seen = true; break; }
            if (!seen) {
                if ((int)p.distinct.size() >= 33) p.capped = true;
                else p.distinct.push_back(v);
            }
        }
    }
    {
        std::lock_guard<std::mutex> lock(g_scanMutex);
        try {
            ScanCacheEntry e;
            e.size = (long long)std::filesystem::file_size(path);
            e.mtime = std::filesystem::last_write_time(path);
            e.scan = out;
            g_scanCache[path] = e;
        } catch (const std::exception&) { /* caching is best-effort */ }
    }
    return out;
}

int csvColumnCount(const std::string& path) {
    TargetScan scan = scanColumns(path, true);
    return scan.cols;
}

LoadedRaw AndGateSource::load(const ExperimentConfig&) const {
    LoadedRaw out;
    out.data = makeAndGate();
    out.targetCol = 0;
    out.name = "AND";
    return out;
}

LoadedRaw OrGateSource::load(const ExperimentConfig&) const {
    LoadedRaw out;
    out.data = makeOrGate();
    out.targetCol = 0;
    out.name = "OR";
    return out;
}

LoadedRaw XorGateSource::load(const ExperimentConfig&) const {
    LoadedRaw out;
    out.data = makeXorGate();
    out.targetCol = 0;
    out.name = "XOR";
    return out;
}

LoadedRaw IrisSource::load(const ExperimentConfig&) const {
    LoadedRaw out;
    out.data = Dataset::loadCSV("data/iris_small.csv", true, 4, 1);
    out.targetCol = 4;
    out.name = "Iris";
    return out;
}

LoadedRaw CsvSource::load(const ExperimentConfig& cfg) const {
    if (cfg.csvPath.empty())
        throw std::runtime_error("No CSV file selected (drop a .csv file onto the window or use Browse).");
    TargetScan scan = scanColumns(cfg.csvPath, cfg.header);
    int col = cfg.targetCol;
    std::string note;
    if (col < 0) {
        if (scan.cols == 0) {
            col = 0; // scan failed; loadCSV below reports the real problem
        } else {
            TargetChoice choice = TargetSelectorChain::defaults().select(scan);
            col = (choice.column >= 0 && choice.column < scan.cols) ? choice.column : scan.cols - 1;
            note = "auto: " + choice.reason;
        }
    } else {
        std::ostringstream ns;
        ns << "selected manually";
        note = ns.str();
        if (scan.cols > 0 && col >= scan.cols) {
            std::ostringstream es;
            es << "Target column " << col << " out of range (file has " << scan.cols << " columns).";
            throw std::runtime_error(es.str());
        }
    }
    if (scan.cols > 0 && col >= 0 && col < scan.cols) {
        const ColumnProfile& prof = scan.columns[(std::size_t)col];
        if (!prof.numeric)
            throw std::runtime_error("Target column " + std::to_string(col) +
                " is non-numeric — pick the label column.");
        if (!prof.capped && prof.distinct.size() <= 1)
            throw std::runtime_error("Target column " + std::to_string(col) + " is constant (all values = " +
                (prof.distinct.empty() ? "blank" : "'" + prof.distinct[0] + "'") +
                ") — predicting a constant is trivially 100% and measures nothing. "
                "Pick the label column.");
    }
    LoadedRaw out;
    out.data = Dataset::loadCSV(cfg.csvPath, cfg.header, col, 1);
    out.targetCol = col;
    out.note = note;
    out.name = "CSV";
    return out;
}

std::unique_ptr<IDatasetSource> DatasetSourceFactory::create(const std::string& key) {
    if (key == "and") return std::make_unique<AndGateSource>();
    if (key == "or")  return std::make_unique<OrGateSource>();
    if (key == "xor") return std::make_unique<XorGateSource>();
    if (key == "iris") return std::make_unique<IrisSource>();
    if (key == "csv") return std::make_unique<CsvSource>();
    throw std::invalid_argument("Unknown dataset '" + key + "' (choose and|or|xor|iris|csv)");
}

} // namespace miniann
