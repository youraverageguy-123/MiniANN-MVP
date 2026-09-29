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
                             const TrainingConfig& cfg) {
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
        if (validation) hist.validationLoss.push_back(lossOn(net_, *validation));

        if (logger_ && (epoch % cfg.logEvery == 0 || epoch == 1 || epoch == cfg.epochs)) {
            std::ostringstream os;
            os << "epoch " << epoch << "/" << cfg.epochs << " train_loss=" << tl;
            if (validation) os << " val_loss=" << hist.validationLoss.back();
            logger_->log(LogLevel::Info, os.str());
        }
    }
    return hist;
}

} // namespace miniann
