#include "miniann/serializer.hpp"
#include "miniann/activation.hpp"
#include "miniann/optimizer.hpp"
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <sstream>

namespace miniann {

// Format: MINIANN 2, "layer n m act[:alpha]", then optional "opt ..." block.
// v1 files (MINIANN 1, plain act names, no opt block) still load.
static std::string actToken(const IActivation& a) {
    if (a.name() == "leaky_relu") {
        const auto* lr = dynamic_cast<const LeakyReLU*>(&a);
        double alpha = lr ? lr->alpha() : 0.01;
        std::ostringstream os;
        os << "leaky_relu:" << std::setprecision(17) << alpha;
        return os.str();
    }
    return a.name();
}

static ActivationPtr actFromToken(const std::string& tok) {
    auto pos = tok.find(':');
    if (pos == std::string::npos) return ActivationFactory::create(tok);
    std::string base = tok.substr(0, pos);
    double alpha = std::stod(tok.substr(pos + 1));
    if (base == "leaky_relu" || base == "leakyrelu") return std::make_shared<LeakyReLU>(alpha);
    throw std::runtime_error("ModelSerializer::load: unknown parameterized activation '" + tok + "'");
}

static void writeOpt(std::ofstream& f, const IOptimizer* opt, const NeuralNetwork& net) {
    if (!opt) return;
    f << std::setprecision(17);
    if (auto* s = dynamic_cast<const SGD*>(opt)) {
        f << "opt sgd " << s->lr() << "\n";
    } else if (auto* m = dynamic_cast<const Momentum*>(opt)) {
        f << "opt momentum " << m->lr() << " " << m->mu() << "\n";
        auto st = m->state();
        for (std::size_t l = 0; l < net.layers().size(); ++l) {
            for (std::size_t j = 0; j < net.layers()[l].neurons().size(); ++j) {
                f << "vw";
                if (l < st.vW.size() && j < st.vW[l].size())
                    for (double v : st.vW[l][j]) f << " " << v;
                else
                    for (std::size_t i = 0; i < net.layers()[l].neurons()[j].weights().size(); ++i) f << " 0";
                f << "\n";
            }
            f << "vb";
            if (l < st.vB.size())
                for (double v : st.vB[l]) f << " " << v;
            else
                for (std::size_t j = 0; j < net.layers()[l].neurons().size(); ++j) f << " 0";
            f << "\n";
        }
    } else if (auto* a = dynamic_cast<const Adam*>(opt)) {
        f << "opt adam " << a->lr() << " " << a->beta1() << " " << a->beta2()
          << " " << a->eps() << " " << a->stepCount() << "\n";
        auto st = a->state();
        for (std::size_t l = 0; l < net.layers().size(); ++l) {
            for (std::size_t j = 0; j < net.layers()[l].neurons().size(); ++j) {
                f << "mw";
                if (l < st.mW.size() && j < st.mW[l].size())
                    for (double v : st.mW[l][j]) f << " " << v;
                else
                    for (std::size_t i = 0; i < net.layers()[l].neurons()[j].weights().size(); ++i) f << " 0";
                f << "\n";
                f << "vw";
                if (l < st.vW.size() && j < st.vW[l].size())
                    for (double v : st.vW[l][j]) f << " " << v;
                else
                    for (std::size_t i = 0; i < net.layers()[l].neurons()[j].weights().size(); ++i) f << " 0";
                f << "\n";
            }
            f << "mb";
            if (l < st.mB.size())
                for (double v : st.mB[l]) f << " " << v;
            else
                for (std::size_t j = 0; j < net.layers()[l].neurons().size(); ++j) f << " 0";
            f << "\n";
            f << "vb";
            if (l < st.vB.size())
                for (double v : st.vB[l]) f << " " << v;
            else
                for (std::size_t j = 0; j < net.layers()[l].neurons().size(); ++j) f << " 0";
            f << "\n";
        }
    }
}

void ModelSerializer::save(const NeuralNetwork& net, const std::string& path) {
    save(net, nullptr, path);
}

void ModelSerializer::save(const NeuralNetwork& net, const IOptimizer* opt,
                           const std::string& path) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("ModelSerializer::save: cannot open " + path);
    f << std::setprecision(17);
    f << "MINIANN 2\n";
    f << "layers " << net.layers().size() << "\n";
    for (auto& layer : net.layers()) {
        if (layer.neurons().empty()) throw std::runtime_error("ModelSerializer::save: empty layer");
        f << "layer " << layer.size() << " " << layer.inputSize() << " " << actToken(layer.neurons()[0].activation()) << "\n";
        for (auto& n : layer.neurons()) {
            for (double w : n.weights()) f << w << " ";
            f << n.bias() << "\n";
        }
    }
    writeOpt(f, opt, net);
}

NeuralNetwork ModelSerializer::load(const std::string& path, std::mt19937& rng) {
    std::unique_ptr<IOptimizer> opt;
    return loadWithOptimizer(path, rng, opt);
}

NeuralNetwork ModelSerializer::loadWithOptimizer(const std::string& path, std::mt19937& rng,
                                                 std::unique_ptr<IOptimizer>& opt) {
    opt.reset();
    std::ifstream f(path);
    if (!f) throw std::runtime_error("ModelSerializer::load: cannot open " + path);
    std::string magic;
    int version;
    if (!(f >> magic >> version)) throw std::runtime_error("ModelSerializer::load: bad header");
    if (magic != "MINIANN") throw std::runtime_error("ModelSerializer::load: bad magic '" + magic + "'");
    if (version != 1 && version != 2) throw std::runtime_error("ModelSerializer::load: unsupported version");
    std::string layersWord;
    std::size_t numLayers;
    if (!(f >> layersWord >> numLayers) || layersWord != "layers")
        throw std::runtime_error("ModelSerializer::load: expected 'layers N'");
    NeuralNetwork net;
    for (std::size_t l = 0; l < numLayers; ++l) {
        std::string lw;
        std::size_t nNeurons, nInputs;
        std::string actName;
        if (!(f >> lw >> nNeurons >> nInputs >> actName) || lw != "layer")
            throw std::runtime_error("ModelSerializer::load: expected 'layer n m act'");
        ActivationPtr act;
        try { act = actFromToken(actName); }
        catch (const std::exception&) {
            throw std::runtime_error("ModelSerializer::load: unknown activation '" + actName + "'");
        }
        Layer layer(nNeurons, nInputs, act, rng);
        for (std::size_t j = 0; j < nNeurons; ++j) {
            Vector w(nInputs);
            for (std::size_t i = 0; i < nInputs; ++i)
                if (!(f >> w[i])) throw std::runtime_error("ModelSerializer::load: truncated weights");
            double b;
            if (!(f >> b)) throw std::runtime_error("ModelSerializer::load: truncated bias");
            layer.neurons()[j].setParameters(w, b);
        }
        net.addLayer(std::move(layer));
    }
    // Optional trailing opt block (v2 only). Absence = weights-only load.
    std::string ow;
    if (!(f >> ow)) return net;
    if (ow != "opt") throw std::runtime_error("ModelSerializer::load: expected 'opt ...'");
    std::string oname;
    if (!(f >> oname)) throw std::runtime_error("ModelSerializer::load: truncated opt block");
    auto readDoubles = [&](std::size_t n, const char* what) {
        Vector v(n);
        for (std::size_t i = 0; i < n; ++i)
            if (!(f >> v[i])) throw std::runtime_error(std::string("ModelSerializer::load: truncated ") + what);
        return v;
    };
    if (oname == "sgd") {
        double lr;
        if (!(f >> lr)) throw std::runtime_error("ModelSerializer::load: truncated opt sgd");
        opt = std::make_unique<SGD>(lr);
    } else if (oname == "momentum") {
        double lr, mu;
        if (!(f >> lr >> mu)) throw std::runtime_error("ModelSerializer::load: truncated opt momentum");
        auto m = std::make_unique<Momentum>(lr, mu);
        Momentum::State st;
        st.vW.resize(net.layers().size());
        st.vB.resize(net.layers().size());
        for (std::size_t l = 0; l < net.layers().size(); ++l) {
            std::size_t nn = net.layers()[l].neurons().size();
            std::size_t ni = net.layers()[l].inputSize();
            st.vW[l].resize(nn);
            for (std::size_t j = 0; j < nn; ++j) {
                std::string tag;
                if (!(f >> tag) || tag != "vw") throw std::runtime_error("ModelSerializer::load: expected 'vw'");
                Vector v = readDoubles(ni, "vw");
                st.vW[l][j] = v;
            }
            std::string tag;
            if (!(f >> tag) || tag != "vb") throw std::runtime_error("ModelSerializer::load: expected 'vb'");
            Vector vb = readDoubles(nn, "vb");
            st.vB[l].assign(vb.begin(), vb.end());
        }
        m->restore(st);
        opt = std::move(m);
    } else if (oname == "adam") {
        double lr, b1, b2, eps;
        std::size_t t;
        if (!(f >> lr >> b1 >> b2 >> eps >> t)) throw std::runtime_error("ModelSerializer::load: truncated opt adam");
        auto a = std::make_unique<Adam>(lr, b1, b2, eps);
        Adam::State st;
        st.t = t;
        st.mW.resize(net.layers().size());
        st.vW.resize(net.layers().size());
        st.mB.resize(net.layers().size());
        st.vB.resize(net.layers().size());
        for (std::size_t l = 0; l < net.layers().size(); ++l) {
            std::size_t nn = net.layers()[l].neurons().size();
            std::size_t ni = net.layers()[l].inputSize();
            st.mW[l].resize(nn);
            st.vW[l].resize(nn);
            for (std::size_t j = 0; j < nn; ++j) {
                std::string tag;
                if (!(f >> tag) || tag != "mw") throw std::runtime_error("ModelSerializer::load: expected 'mw'");
                st.mW[l][j] = readDoubles(ni, "mw");
                if (!(f >> tag) || tag != "vw") throw std::runtime_error("ModelSerializer::load: expected 'vw'");
                st.vW[l][j] = readDoubles(ni, "vw");
            }
            std::string tag;
            if (!(f >> tag) || tag != "mb") throw std::runtime_error("ModelSerializer::load: expected 'mb'");
            Vector mb = readDoubles(nn, "mb");
            st.mB[l].assign(mb.begin(), mb.end());
            if (!(f >> tag) || tag != "vb") throw std::runtime_error("ModelSerializer::load: expected 'vb'");
            Vector vb = readDoubles(nn, "vb");
            st.vB[l].assign(vb.begin(), vb.end());
        }
        a->restore(st);
        opt = std::move(a);
    } else {
        throw std::runtime_error("ModelSerializer::load: unknown opt '" + oname + "'");
    }
    return net;
}

void CSVLossExporter::exportHistory(const TrainingHistory& hist, const std::string& filepath) {
    std::ofstream f(filepath);
    if (!f) throw std::runtime_error("CSVLossExporter: cannot open " + filepath);
    f << "epoch,train_loss,val_loss,train_acc,val_acc\n";
    f << std::setprecision(17);
    for (std::size_t i = 0; i < hist.trainLoss.size(); ++i) {
        f << (i + 1) << "," << hist.trainLoss[i] << ",";
        if (i < hist.validationLoss.size()) f << hist.validationLoss[i];
        f << ",";
        if (i < hist.trainAcc.size()) f << hist.trainAcc[i];
        f << ",";
        if (i < hist.validationAcc.size()) f << hist.validationAcc[i];
        f << "\n";
    }
}

} // namespace miniann
