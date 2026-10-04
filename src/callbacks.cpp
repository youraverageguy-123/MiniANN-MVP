#include "miniann/callbacks.hpp"
#include <stdexcept>

namespace miniann {

// ---------------------------------------------------------------- EarlyStopping

EarlyStopping::EarlyStopping(int patience, double minDelta)
    : patience_(patience), minDelta_(minDelta) {
    if (patience < 1)
        throw std::invalid_argument("EarlyStopping: patience must be >= 1");
    if (!(minDelta >= 0.0))   // also rejects NaN
        throw std::invalid_argument("EarlyStopping: minDelta must be >= 0");
}

void EarlyStopping::onEpoch(int epoch, const TrainingHistory& hist) {
    if (stopped_) return;                       // decision already made
    // Watch validation loss if there is one (it detects overfitting),
    // otherwise fall back to training loss.
    usedValidation_ = !hist.validationLoss.empty();
    const std::vector<double>& watched =
        usedValidation_ ? hist.validationLoss : hist.trainLoss;
    if (watched.empty()) return;
    const double current = watched.back();

    if (current < best_ - minDelta_) {          // real improvement
        best_ = current;
        bestEpoch_ = epoch;
        wait_ = 0;
    } else {                                    // no improvement (NaN lands here too)
        ++wait_;
        if (wait_ >= patience_) {
            stopped_ = true;
            stoppedAt_ = epoch;
        }
    }
}

void EarlyStopping::reset() {
    best_ = std::numeric_limits<double>::infinity();
    bestEpoch_ = 0;
    wait_ = 0;
    stopped_ = false;
    stoppedAt_ = 0;
    usedValidation_ = false;
}

// ----------------------------------------------------------------- CallbackList

void CallbackList::add(TrainingCallback* cb) {
    if (!cb) throw std::invalid_argument("CallbackList::add: null callback");
    items_.push_back(cb);
}

void CallbackList::onEpoch(int epoch, const TrainingHistory& hist) {
    for (TrainingCallback* cb : items_) cb->onEpoch(epoch, hist);
}

bool CallbackList::shouldStop() const {
    for (const TrainingCallback* cb : items_)
        if (cb->shouldStop()) return true;
    return false;
}

} // namespace miniann
