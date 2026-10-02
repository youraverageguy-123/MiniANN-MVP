#include "miniann/serializer.hpp"
#include "miniann/activation.hpp"
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <sstream>

namespace miniann {

void ModelSerializer::save(const NeuralNetwork& net, const std::string& path) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("ModelSerializer::save: cannot open " + path);
    f << std::setprecision(17);
    f << "MINIANN 1\n";
    f << "layers " << net.layers().size() << "\n";
    for (auto& layer : net.layers()) {
        if (layer.neurons().empty()) throw std::runtime_error("ModelSerializer::save: empty layer");
        std::string act = layer.neurons()[0].activation().name();
        f << "layer " << layer.size() << " " << layer.inputSize() << " " << act << "\n";
        for (auto& n : layer.neurons()) {
            for (double w : n.weights()) f << w << " ";
            f << n.bias() << "\n";
        }
    }
}

NeuralNetwork ModelSerializer::load(const std::string& path, std::mt19937& rng) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("ModelSerializer::load: cannot open " + path);
    std::string magic;
    int version;
    if (!(f >> magic >> version)) throw std::runtime_error("ModelSerializer::load: bad header");
    if (magic != "MINIANN") throw std::runtime_error("ModelSerializer::load: bad magic '" + magic + "'");
    if (version != 1) throw std::runtime_error("ModelSerializer::load: unsupported version");
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
        try { act = ActivationFactory::create(actName); }
        catch (...) { throw std::runtime_error("ModelSerializer::load: unknown activation '" + actName + "'"); }
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
