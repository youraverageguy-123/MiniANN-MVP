#pragma once
// Training callbacks built on the existing TrainingCallback interface.
//
//   EarlyStopping : stops training automatically once the loss stops improving.
//   CallbackList  : Composite -- lets several callbacks share the single
//                   callback slot of Trainer::fit() (e.g. LiveConsole + EarlyStopping).
#include "miniann/trainer.hpp"
#include <limits>
#include <vector>

namespace miniann {

class EarlyStopping : public TrainingCallback {
public:
    // patience : how many epochs in a row without improvement we tolerate.
    // minDelta : how much the loss must drop to count as an "improvement".
    // Throws std::invalid_argument for patience < 1 or minDelta < 0.
    explicit EarlyStopping(int patience = 50, double minDelta = 1e-6);

    // Called by the Trainer at the end of every epoch.
    void onEpoch(int epoch, const TrainingHistory& hist) override;
    // Polled by the Trainer right after onEpoch(): true => break the loop.
    bool shouldStop() const override { return stopped_; }

    // Forget everything so the same object can watch a new training run.
    void reset();

    int patience() const { return patience_; }
    double minDelta() const { return minDelta_; }
    bool stopped() const { return stopped_; }
    int stoppedAtEpoch() const { return stoppedAt_; }   // 0 if never stopped
    int bestEpoch() const { return bestEpoch_; }        // epoch with lowest loss
    double bestLoss() const { return best_; }
    bool usedValidation() const { return usedValidation_; }

private:
    int patience_;
    double minDelta_;
    double best_ = std::numeric_limits<double>::infinity();
    int bestEpoch_ = 0;
    int wait_ = 0;          // epochs since the last improvement
    bool stopped_ = false;
    int stoppedAt_ = 0;
    bool usedValidation_ = false;
};

class CallbackList : public TrainingCallback {
public:
    // Non-owning: the caller keeps the callbacks alive during fit().
    // Throws std::invalid_argument on nullptr.
    void add(TrainingCallback* cb);
    std::size_t size() const { return items_.size(); }

    // Forwards the epoch event to EVERY callback.
    void onEpoch(int epoch, const TrainingHistory& hist) override;
    // Stops if ANY callback asks to stop.
    bool shouldStop() const override;

private:
    std::vector<TrainingCallback*> items_;
};

} // namespace miniann
