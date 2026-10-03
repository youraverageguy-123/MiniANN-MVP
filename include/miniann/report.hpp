#pragma once
// MINIANN REPORT SECTIONS (CLI‑side, zero‑conflict with raylib visualizer)
// Uses the IHtmlSection class defined in visualizer.hpp.
// Provides concrete sections that can be embedded in HtmlReport via addSection().

#include "miniann/cli.hpp"
#include "miniann/visualizer.hpp"
#include "miniann/network.hpp"
#include "miniann/dataset.hpp"
#include "miniann/trainer.hpp"
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

namespace miniann {

// Escape text for embedding inside HTML <pre> cards.
inline std::string htmlEscapeText(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            default: out += c; break;
        }
    }
    return out;
}

// -------------------------------------------------------------------
// ConfigSection – dumps the exact CLI invocation and every knob.
// -------------------------------------------------------------------
class ConfigSection : public IHtmlSection {
public:
    // Stores copies (not refs/argv pointers) so the section stays valid
    // even after the caller's Options goes out of scope.
    ConfigSection(const cli::Options& o, int argc, char** argv,
                  std::string defaultReport = "xor_report.html")
        : o_(o), cmdLine_(cli::commandLine(argc, argv)),
          defaultReport_(std::move(defaultReport)) {}

    std::string toHtml() const override {
        return "<div class=\"card\"><h2>run config (CLI)</h2>\n<pre>"
               + htmlEscapeText(miniann::cli::describe(o_, cmdLine_, defaultReport_))
               + "</pre></div>\n";
    }

private:
    cli::Options o_;
    std::string cmdLine_;
    std::string defaultReport_;
};

// -------------------------------------------------------------------
// SummarySection – one‑line result table (final loss, accuracy, epochs).
// -------------------------------------------------------------------
class SummarySection : public IHtmlSection {
public:
    // accuracy is a fraction 0..1 (not percent); rendered as %.
    SummarySection(double finalLoss, double accuracy01, int epochs,
                   const std::string& optimizer, double lr)
        : loss_(finalLoss), acc_(accuracy01), epochs_(epochs),
          opt_(optimizer), lr_(lr) {}

    std::string toHtml() const override {
        std::ostringstream os;
        os << std::setprecision(6) << std::fixed;
        os << "<div class=\"card\"><h2>result summary</h2>\n"
           << "<table border=\"1\" cellpadding=\"6\" cellspacing=\"0\">\n"
           << "<tr><th>optimizer</th><th>lr</th><th>epochs</th>"
              "<th>final loss</th><th>accuracy</th></tr>\n<tr><td>"
           << htmlEscapeText(opt_) << "</td><td>" << lr_ << "</td><td>" << epochs_
           << "</td><td>" << loss_ << "</td><td>" << (acc_ * 100.0) << "%</td></tr>\n"
           << "</table></div>\n";
        return os.str();
    }

private:
    double loss_, acc_;
    int epochs_;
    std::string opt_;
    double lr_;
};

// -------------------------------------------------------------------
// HtmlReportBuilder – fluent builder that populates an HtmlReport.
// -------------------------------------------------------------------
class HtmlReportBuilder {
public:
    explicit HtmlReportBuilder(std::string title) : report_(std::move(title)) {}

    HtmlReportBuilder& withSeries(const std::string& name,
                                  const TrainingHistory& h,
                                  const std::string& color = "") {
        ReportSeries rs;
        rs.name = name;
        rs.color = color;
        rs.loss = h.trainLoss;
        rs.acc = h.trainAcc;
        report_.addSeries(rs);
        return *this;
    }

    HtmlReportBuilder& withBoundary(NeuralNetwork& net, const Dataset& data) {
        report_.setBoundary(&net, &data);
        return *this;
    }

    HtmlReportBuilder& withConfig(const cli::Options& o, int argc, char** argv,
                                  const std::string& defaultReport = "xor_report.html") {
        report_.addSection(std::make_unique<ConfigSection>(o, argc, argv, defaultReport));
        return *this;
    }

    HtmlReportBuilder& withSummary(double finalLoss, double acc, int epochs,
                                   const std::string& opt, double lr) {
        report_.addSection(std::make_unique<SummarySection>(finalLoss, acc, epochs, opt, lr));
        return *this;
    }

    HtmlReportBuilder& withPre(const std::string& heading, const std::string& body) {
        report_.addPre(heading, body);
        return *this;
    }

    HtmlReport build() { return std::move(report_); }

private:
    HtmlReport report_;
};

} // namespace miniann