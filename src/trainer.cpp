#include "miniann/trainer.hpp"
#include <random>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace miniann {

Trainer::Trainer(NeuralNetwork& net, const ILoss& loss, IOptimizer& opt, ILogger* logger)
    : net_(net), loss_(loss), opt_(opt), logger_(logger) {}

TrainingHistory Trainer::fit(const Dataset& train, const Dataset* validation,
                             const TrainingConfig& cfg, TrainingCallback* cb) {
    train.validate();
    if (validation) validation->validate();
    TrainingHistory hist;
    std::mt19937 rng(cfg.seed);

    auto lossOn = [&](NeuralNetwork& n, const Dataset& d) {
        double s = 0.0;
        for (std::size_t i = 0; i < d.size(); ++i)
            s += loss_.compute(n.predict(d.input(i)), d.target(i));
        return d.size() ? s / double(d.size()) : 0.0;
    };

    // Classification accuracy: 0.5 threshold for 1-output nets, argmax otherwise.
    auto accOn = [&](NeuralNetwork& n, const Dataset& d) {
        if (d.size() == 0) return 0.0;
        std::size_t correct = 0;
        for (std::size_t i = 0; i < d.size(); ++i) {
            Vector p = n.predict(d.input(i));
            const Vector& t = d.target(i);
            bool ok = false;
            if (p.size() == 1) {
                ok = ((p[0] >= 0.5) ? 1 : 0) == ((t[0] >= 0.5) ? 1 : 0);
            } else if (p.size() == t.size()) {
                std::size_t bp = 0, bt = 0;
                for (std::size_t k = 1; k < p.size(); ++k) {
                    if (p[k] > p[bp]) bp = k;
                    if (t[k] > t[bt]) bt = k;
                }
                ok = (bp == bt);
            }
            if (ok) ++correct;
        }
        return double(correct) / double(d.size());
    };

    for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
        // index order
        std::vector<std::size_t> idx(train.size());
        std::iota(idx.begin(), idx.end(), 0);
        if (cfg.shuffle) std::shuffle(idx.begin(), idx.end(), rng);

        std::size_t B = (cfg.batchSize == 0) ? train.size() : cfg.batchSize;
        if (B == 0) B = train.size();

        for (std::size_t b = 0; b < idx.size(); b += B) {
            std::size_t bend = std::min(idx.size(), b + B);
            std::size_t bs = bend - b;
            net_.zeroGradients();
            for (std::size_t k = b; k < bend; ++k) {
                std::size_t si = idx[k];
                Vector pred = net_.predict(train.input(si));
                net_.backward(loss_.gradient(pred, train.target(si)));
            }
            opt_.step(net_, bs);
        }

        double tl = lossOn(net_, train);
        hist.trainLoss.push_back(tl);
        hist.trainAcc.push_back(accOn(net_, train));
        if (validation) {
            hist.validationLoss.push_back(lossOn(net_, *validation));
            hist.validationAcc.push_back(accOn(net_, *validation));
        }

        if (logger_ && (epoch % cfg.logEvery == 0 || epoch == 1 || epoch == cfg.epochs)) {
            std::ostringstream os;
            os << "epoch " << epoch << "/" << cfg.epochs << " train_loss=" << tl
               << " train_acc=" << hist.trainAcc.back();
            if (validation) os << " val_loss=" << hist.validationLoss.back()
                               << " val_acc=" << hist.validationAcc.back();
            logger_->log(LogLevel::Info, os.str());
        }
        if (cb) cb->onEpoch(epoch, hist);
    }
    return hist;
}

} // namespace miniann
