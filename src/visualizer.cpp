#include "miniann/visualizer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace miniann {

namespace {
// Compact tick label: 0.25, 100, 1.2e-05 ...
std::string tick(double v) {
    char buf[32];
    if (v == 0.0) return "0";
    double a = std::abs(v);
    if (a >= 10000.0 || a < 0.01) std::snprintf(buf, sizeof buf, "%.1e", v);
    else if (a >= 100.0) std::snprintf(buf, sizeof buf, "%.0f", v);
    else if (a >= 1.0) std::snprintf(buf, sizeof buf, "%.2f", v);
    else std::snprintf(buf, sizeof buf, "%.2g", v);
    return buf;
}

// True line plot: y-axis with 5 tick labels, x-axis with 1..N labels,
// '*' polyline (vertical runs joined with '|'). Log-y when span > 50x.
std::string linePlot(const std::vector<double>& ys, int width, int height,
                     bool logY, bool fixZeroOne, const std::string& caption) {
    std::ostringstream os;
    if (ys.empty() || width < 10 || height < 4) return "(no data)\n";
    std::size_t n = ys.size();

    double lo = *std::min_element(ys.begin(), ys.end());
    double hi = *std::max_element(ys.begin(), ys.end());
    if (fixZeroOne) { lo = 0.0; hi = 1.0; }
    if (!(hi > lo)) { hi = lo + 1.0; if (!fixZeroOne) lo -= 1.0; }
    bool useLog = logY && lo > 0.0 && hi / lo > 50.0;
    double llo = useLog ? std::log10(lo) : lo;
    double lhi = useLog ? std::log10(hi) : hi;

    // one value per column (average per bucket)
    std::vector<double> col;
    col.assign(std::size_t(width), 0.0);
    for (int c = 0; c < width; ++c) {
        std::size_t a = n * std::size_t(c) / std::size_t(width);
        std::size_t b = n * std::size_t(c + 1) / std::size_t(width);
        if (b <= a) b = a + 1;
        double s = 0.0;
        for (std::size_t k = a; k < b; ++k) s += ys[k];
        col[std::size_t(c)] = s / double(b - a);
    }
    auto rowOf = [&](double v) {
        double t = useLog ? std::log10(std::max(v, 1e-300)) : v;
        double f = (lhi - t) / (lhi - llo); // 0=top
        int r = int(std::lround(f * double(height - 1)));
        return std::max(0, std::min(height - 1, r));
    };

    std::vector<std::string> grid(std::size_t(height), std::string(std::size_t(width), ' '));
    std::vector<int> rows;
    rows.assign(std::size_t(width), 0);
    for (int c = 0; c < width; ++c) rows[std::size_t(c)] = rowOf(col[std::size_t(c)]);
    for (int c = 0; c < width; ++c) grid[std::size_t(rows[std::size_t(c)])][std::size_t(c)] = '*';
    for (int c = 1; c < width; ++c) { // join vertical runs
        int r0 = rows[std::size_t(c - 1)], r1 = rows[std::size_t(c)];
        for (int r = std::min(r0, r1); r <= std::max(r0, r1); ++r)
            if (grid[std::size_t(r)][std::size_t(c)] == ' ') grid[std::size_t(r)][std::size_t(c)] = '|';
    }

    const int YW = 9;
    for (int r = 0; r < height; ++r) {
        std::string lab = " ";
        if (r % ((height - 1) / 4 == 0 ? 1 : (height - 1) / 4) == 0 || r == height - 1) {
            double t = lhi - (lhi - llo) * double(r) / double(height - 1);
            double v = useLog ? std::pow(10.0, t) : t;
            lab = tick(v);
        }
        os << std::string(std::size_t(std::max(0, YW - 1 - int(lab.size()))), ' ')
           << lab << '|' << grid[std::size_t(r)] << "\n";
    }
    os << std::string(std::size_t(YW), ' ') << '+'
       << std::string(std::size_t(width), '-') << "\n";
    os << std::string(std::size_t(YW + 1), ' ') << "1"
       << std::string(std::size_t(width > 20 ? width - 20 : 0), ' ')
       << n << "  (" << caption << (useLog ? ", log scale" : "") << ")\n";
    return os.str();
}
} // namespace

std::string LossCurve::render() {
    if (hist_.trainLoss.empty()) return "(no history)\n";
    return linePlot(hist_.trainLoss, width_, height_, true, false, "loss vs epoch");
}

BoundaryMap::BoundaryMap(NeuralNetwork& net, const Dataset& data, int width, int height)
    : net_(net), data_(data), width_(width), height_(height) {}

std::string BoundaryMap::render() {
    std::ostringstream os;
    if (data_.size() == 0) return "(no data)\n";
    // grid over unit square with small margin; '#' = class 1, '.' = class 0
    std::vector<std::string> grid(std::size_t(height_), std::string(std::size_t(width_), ' '));
    for (int r = 0; r < height_; ++r) {
        double x2 = 1.1 - 1.2 * double(r) / double(height_ - 1);
        for (int c = 0; c < width_; ++c) {
            double x1 = -0.1 + 1.2 * double(c) / double(width_ - 1);
            Vector p = net_.predict({x1, x2});
            grid[std::size_t(r)][std::size_t(c)] = (p[0] >= 0.5 ? '#' : '.');
        }
    }
    // stamp training points as 0/1 so correctness is visible at a glance
    if (!data_.input(0).empty() && data_.input(0).size() == 2 && data_.target(0).size() == 1) {
        for (std::size_t i = 0; i < data_.size(); ++i) {
            int c = int(std::lround((data_.input(i)[0] + 0.1) / 1.2 * double(width_ - 1)));
            int r = int(std::lround((1.1 - data_.input(i)[1]) / 1.2 * double(height_ - 1)));
            c = std::max(0, std::min(width_ - 1, c));
            r = std::max(0, std::min(height_ - 1, r));
            grid[std::size_t(r)][std::size_t(c)] = (data_.target(i)[0] >= 0.5 ? '1' : '0');
        }
    }
    for (int r = 0; r < height_; ++r)
        os << grid[std::size_t(r)] << " "
           << (r % 5 == 0 ? std::to_string(1.1 - 1.2 * double(r) / double(height_ - 1)).substr(0, 4) : "")
           << "\n";
    os << "key: #=predicts 1, .=predicts 0, 0/1=training points\n";
    return os.str();
}

std::string AccuracyCurve::render() {
    if (hist_.trainAcc.empty()) return "(no accuracy history)\n";
    return linePlot(hist_.trainAcc, width_, height_, false, true, "accuracy vs epoch");
}

std::string NetworkGraph::render() {
    std::ostringstream os;
    if (net_.layers().empty()) return "(empty network)\n";
    os << "in(" << net_.layers()[0].neurons()[0].weights().size() << ")";
    for (auto& layer : net_.layers()) {
        const auto& ns = layer.neurons();
        std::string act = ns.empty() ? "?" : ns[0].activation().name();
        os << " ==> [" << ns.size() << " x " << act << "]";
    }
    os << " => out\n";
    std::size_t li = 0;
    for (auto& layer : net_.layers()) {
        os << "  L" << li++ << ": in=" << layer.neurons()[0].weights().size()
           << " neurons=" << layer.neurons().size()
           << " act=" << layer.neurons()[0].activation().name() << "\n";
    }
    return os.str();
}

std::string WeightsTable::render() {
    std::ostringstream os;
    os << std::setprecision(4) << std::fixed;
    std::size_t li = 0;
    for (auto& layer : net_.layers()) {
        std::size_t ji = 0;
        for (auto& n : layer.neurons()) {
            os << "L" << li << "N" << ji++ << " w=[";
            const Vector& w = n.weights();
            for (std::size_t i = 0; i < w.size(); ++i)
                os << (i ? "," : "") << w[i];
            os << "] b=" << n.bias() << "\n";
        }
        ++li;
    }
    return os.str();
}

namespace {
const char* kPalette[] = {"#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"};

std::string tickG(double v) { // %.4g-style label for SVG ticks
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.4g", v);
    return buf;
}

std::string htmlEscape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '&') o += "&amp;";
        else if (c == '<') o += "&lt;";
        else if (c == '>') o += "&gt;";
        else o += c;
    }
    return o;
}

std::string f2(double v) {
    char b[32];
    std::snprintf(b, sizeof b, "%.2f", v);
    return b;
}

// Downsampled SVG polyline chart. logY for loss spans, fixed 0..1 for accuracy.
std::string svgChart(const std::vector<std::pair<std::string, std::vector<double>>>& named,
                     const std::vector<std::string>& colors, bool logY,
                     const std::string& heading, const std::string& yName) {
    const int W = 640, H = 260, ML = 70, MR = 14, MT = 34, MB = 42;
    std::ostringstream os;
    os << "<h2>" << htmlEscape(heading) << "</h2>\n";
    if (named.empty()) return "<p>(no data)</p>\n";

    double lo = 0.0, hi = 1.0;
    if (logY) {
        lo = hi = -1.0;
        bool first = true;
        for (auto& s : named)
            for (double v : s.second) {
                double lv = std::log10(std::max(v, 1e-300));
                if (first) { lo = hi = lv; first = false; }
                else { lo = std::min(lo, lv); hi = std::max(hi, lv); }
            }
        if (!(hi > lo)) hi = lo + 1.0;
    }
    std::size_t n = 0;
    for (auto& s : named) n = std::max(n, s.second.size());
    if (n == 0) return "<p>(no data)</p>\n";
    const std::size_t maxPts = 320;

    auto X = [&](std::size_t k) {
        return ML + (W - ML - MR) * (n < 2 ? 0.0 : double(k) / double(n - 1));
    };
    auto Y = [&](double v) {
        double t = logY ? std::log10(std::max(v, 1e-300)) : v;
        double f = (hi - t) / (hi - lo);
        return MT + (H - MT - MB) * std::max(0.0, std::min(1.0, f));
    };

    os << "<svg width=\"" << W << "\" height=\"" << H << "\" style=\"background:#fff;border:1px solid #ccc\">\n";
    for (int g = 0; g <= 4; ++g) { // gridlines + y ticks
        double t = hi - (hi - lo) * double(g) / 4.0;
        double y = MT + (H - MT - MB) * double(g) / 4.0;
        double v = logY ? std::pow(10.0, t) : t;
        os << "<line x1=\"" << ML << "\" y1=\"" << f2(y) << "\" x2=\"" << (W - MR)
           << "\" y2=\"" << f2(y) << "\" stroke=\"#eee\"/>\n";
        os << "<text x=\"" << (ML - 6) << "\" y=\"" << f2(y + 4)
           << "\" text-anchor=\"end\" font-size=\"11\" fill=\"#555\">" << tickG(v) << "</text>\n";
    }
    for (int g = 0; g <= 2; ++g) { // x ticks: epoch 1 .. N
        std::size_t k = n * std::size_t(g) / 2;
        if (k > 0) --k;
        double x = X(std::min(k, n - 1));
        os << "<text x=\"" << f2(x) << "\" y=\"" << (H - MB + 18)
           << "\" text-anchor=\"middle\" font-size=\"11\" fill=\"#555\">" << (std::min(k, n - 1) + 1) << "</text>\n";
    }
    os << "<text x=\"" << (W - MR) << "\" y=\"" << (H - 8)
       << "\" text-anchor=\"end\" font-size=\"11\" fill=\"#555\">epoch</text>\n";
    os << "<text x=\"10\" y=\"16\" font-size=\"12\" fill=\"#333\">" << htmlEscape(yName)
       << (logY ? " (log scale)" : "") << "</text>\n";

    for (std::size_t s = 0; s < named.size(); ++s) {
        const auto& ys = named[s].second;
        if (ys.empty()) continue;
        os << "<polyline fill=\"none\" stroke=\"" << colors[s % 5]
           << "\" stroke-width=\"2\" points=\"";
        for (std::size_t i = 0; i < maxPts; ++i) {
            std::size_t k = ys.size() * i / maxPts;
            if (k >= ys.size()) k = ys.size() - 1;
            std::size_t epoch = ys.size() * i / maxPts;
            double x = X(std::min(epoch, n - 1));
            os << f2(x) << "," << f2(Y(ys[k])) << " ";
        }
        os << f2(X(n - 1)) << "," << f2(Y(ys.back())) << "\"/>\n";
    }
    double lx = double(W - MR);
    for (std::size_t s = named.size(); s-- > 0;) { // legend, top-right
        lx -= 12.0 + double(named[s].first.size()) * 6.6 + 14.0;
        os << "<rect x=\"" << f2(lx) << "\" y=\"8\" width=\"10\" height=\"10\" fill=\""
           << colors[s % 5] << "\"/><text x=\"" << f2(lx + 12) << "\" y=\"17\" font-size=\"11\">"
           << htmlEscape(named[s].first) << "</text>\n";
    }
    os << "</svg>\n";
    return os.str();
}

} // namespace

void HtmlReport::addPre(const std::string& heading, const std::string& text) {
    pres_.emplace_back(heading, text);
}

std::string JsModelExporter::exportModelJson(const NeuralNetwork& net) {
    std::ostringstream os;
    os << std::setprecision(17);
    os << "{\"layers\":[";
    bool firstL = true;
    for (auto& layer : net.layers()) {
        if (!firstL) os << ",";
        firstL = false;
        std::string act = layer.neurons().empty() ? "sigmoid"
                                                  : layer.neurons()[0].activation().name();
        os << "{\"act\":\"" << act << "\",\"neurons\":[";
        bool firstN = true;
        for (auto& n : layer.neurons()) {
            if (!firstN) os << ",";
            firstN = false;
            os << "{\"w\":[";
            const Vector& w = n.weights();
            for (std::size_t i = 0; i < w.size(); ++i) os << (i ? "," : "") << w[i];
            os << "],\"b\":" << n.bias() << "}";
        }
        os << "]}";
    }
    os << "]}";
    return os.str();
}

std::string JsModelExporter::exportPointsJson(const Dataset& data) {
    std::ostringstream os;
    os << std::setprecision(17) << "[";
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (i) os << ",";
        os << "{\"x\":[";
        const Vector& x = data.input(i);
        for (std::size_t k = 0; k < x.size(); ++k) os << (k ? "," : "") << x[k];
        os << "],\"t\":[";
        const Vector& t = data.target(i);
        for (std::size_t k = 0; k < t.size(); ++k) os << (k ? "," : "") << t[k];
        os << "]}";
    }
    os << "]";
    return os.str();
}

std::string HtmlReport::buildProbeSection() const {
    if (!bnet_ || bnet_->layers().empty()) return "<p>(no model embedded)</p>\n";
    std::size_t inDim = bnet_->layers()[0].neurons()[0].weights().size();
    std::size_t outDim = bnet_->layers().back().neurons().size();
    bool is2d = (inDim == 2 && outDim == 1);
    if (bdata_ && bdata_->size() > 0) {
        is2d = (bdata_->input(0).size() == 2 && bdata_->target(0).size() == 1);
    }
    std::string modelJson = JsModelExporter::exportModelJson(*bnet_);
    std::string pointsJson = (bdata_ && bdata_->size() > 0)
                                 ? JsModelExporter::exportPointsJson(*bdata_)
                                 : "[]";

    std::ostringstream os;
    os << "<div class=\"card\"><h2>Try your own input (live, no re-run)</h2>\n";
    os << "<p class=\"muted\">Move the sliders or type numbers — prediction, neuron firing "
          "and the probe dot update instantly in your browser. Same math as the C++ "
          "<code>predict()</code>.</p>\n";
    os << "<div id=\"inputs\"></div>\n";
    if (is2d) {
        os << "<label>decision threshold <span id=\"thVal\">0.50</span>\n"
           << "<input id=\"th\" type=\"range\" min=\"0.05\" max=\"0.95\" step=\"0.01\" value=\"0.5\"></label>\n";
        os << "<canvas id=\"bmap\" width=\"492\" height=\"252\"></canvas>\n"
           << "<p class=\"muted\">blue = predicts 1, light = predicts 0, numbered dots = training "
              "points, red ring = your probe. Drag the threshold to recolor the map live.</p>\n";
    }
    os << "<div id=\"pred\" class=\"pred\">–</div>\n";
    os << "<h3>Neuron firing (mirrors CLI live panel)</h3><div id=\"fire\"></div>\n";
    os << "<script>\nconst MODEL=" << modelJson << ";\nconst POINTS=" << pointsJson << ";\n";
    os << R"JS(
const IN_DIM=MODEL.layers[0].neurons[0].w.length;
const OUT_DIM=MODEL.layers[MODEL.layers.length-1].neurons.length;
function act(n,z){if(n==="sigmoid")return 1/(1+Math.exp(-z));if(n==="tanh")return Math.tanh(z);
if(n==="relu")return z>0?z:0;if(n==="leaky_relu"||n==="leakyrelu")return z>0?z:0.01*z;
if(n==="swish")return z/(1+Math.exp(-z));return 1/(1+Math.exp(-z));}
function forward(x){let a=x.slice(),trace=[];for(const L of MODEL.layers){let n=[];for(const u of L.neurons){
let z=u.b;for(let i=0;i<a.length;i++)z+=u.w[i]*a[i];let o=act(L.act,z);n.push(o);}trace.push(n);a=n;}return{out:a,trace};}
function clsOf(o){if(o.length===1)return o[0]>=state.th?1:0;let b=0;for(let k=1;k<o.length;k++)if(o[k]>o[b])b=k;return b;}
function confOf(o,c){if(o.length===1)return c?o[0]:1-o[0];return o[c];}
const state={x:Array(IN_DIM).fill(0.5),th:0.5};
function buildInputs(){const d=document.getElementById('inputs');d.innerHTML='';
for(let i=0;i<IN_DIM;i++){const w=document.createElement('div');w.className='row';
w.innerHTML='<label>x'+(i+1)+'</label><input id="s'+i+'" type="range" min="-0.5" max="1.5" step="0.01" value="'+state.x[i]+'">'
+'<input id="n'+i+'" type="number" min="-0.5" max="1.5" step="0.01" value="'+state.x[i]+'">';
d.appendChild(w);}
for(let i=0;i<IN_DIM;i++){const s=document.getElementById('s'+i),n=document.getElementById('n'+i);
s.oninput=()=>{state.x[i]=parseFloat(s.value);n.value=s.value;update();};
n.oninput=()=>{let v=Math.max(-0.5,Math.min(1.5,parseFloat(n.value)||0));state.x[i]=v;s.value=v;update();};}}
function fireHtml(tr){let h='';for(let l=0;l<tr.length;l++){h+='<div class="lyr">L'+l+' ';
for(let j=0;j<tr[l].length;j++){const v=tr[l][j];const pct=Math.max(0,Math.min(100,Math.round(v*100)));
h+='<span class="nb" title="L'+l+'N'+j+'='+v.toFixed(3)+'"><i style="width:'+pct+'%"></i><b>L'+l+'N'+j+' '+v.toFixed(2)+'</b></span>';}h+='</div>';}return h;}
const cv=document.getElementById('bmap');
function drawMap(){if(!cv||IN_DIM!==2||OUT_DIM!==1)return;const GW=41,GH=21,W=cv.width,H=cv.height;
const ctx=cv.getContext('2d'),cw=W/GW,ch=H/GH;
for(let r=0;r<GH;r++){for(let c=0;c<GW;c++){const x1=-0.1+1.2*c/(GW-1),x2=1.1-1.2*r/(GH-1);
const p=forward([x1,x2]).out[0];ctx.fillStyle=p>=state.th?'#7fb3d5':'#f0f0f0';ctx.fillRect(c*cw,r*ch,cw+1,ch+1);}}
for(const p of POINTS){if(p.x.length!==2)continue;const cx=(p.x[0]+0.1)/1.2*W,cy=(1.1-p.x[1])/1.2*H;
ctx.beginPath();ctx.arc(cx,cy,8,0,7);ctx.fillStyle=p.t[0]>=0.5?'#08519c':'#fff';ctx.fill();ctx.strokeStyle='#000';ctx.stroke();
ctx.fillStyle=p.t[0]>=0.5?'#fff':'#000';ctx.font='9px sans-serif';ctx.textAlign='center';
ctx.fillText(p.t[0]>=0.5?'1':'0',cx,cy+3);}
const px=(state.x[0]+0.1)/1.2*W,py=(1.1-state.x[1])/1.2*H;
ctx.beginPath();ctx.arc(px,py,9,0,7);ctx.strokeStyle='#e74c3c';ctx.lineWidth=3;ctx.stroke();}
function update(){const r=forward(state.x),c=clsOf(r.out),cf=confOf(r.out,c);
document.getElementById('pred').textContent='class '+c+'  confidence '+(cf*100).toFixed(1)+'%   raw ['+r.out.map(v=>v.toFixed(4)).join(', ')+']';
document.getElementById('fire').innerHTML=fireHtml(r.trace);drawMap();}
buildInputs();
const th=document.getElementById('th');
if(th)th.oninput=()=>{state.th=parseFloat(th.value);document.getElementById('thVal').textContent=state.th.toFixed(2);update();};
update();
)JS";
    os << "</script></div>\n";
    return os.str();
}

std::string HtmlReport::render() {
    std::vector<std::pair<std::string, std::vector<double>>> losses, accs;
    std::vector<std::string> colors;
    for (std::size_t i = 0; i < series_.size(); ++i) {
        colors.push_back(series_[i].color.empty() ? kPalette[i % 5] : series_[i].color);
        losses.emplace_back(series_[i].name, series_[i].loss);
        if (!series_[i].acc.empty()) accs.emplace_back(series_[i].name, series_[i].acc);
    }
    std::ostringstream os;
    os << "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
          "content=\"width=device-width,initial-scale=1\"><title>" << htmlEscape(title_)
       << "</title>\n<style>body{font-family:system-ui,sans-serif;max-width:760px;margin:24px auto;"
          "padding:0 12px;color:#222}h1{font-size:22px}h2{font-size:17px;margin:18px 0 8px}"
          ".card{border:1px solid #ddd;border-radius:10px;padding:14px;margin:16px 0;background:#fcfcfc}"
          "pre{background:#f5f5f5;padding:12px;overflow-x:auto;border-radius:8px;font-size:12px}"
          ".muted{color:#666;font-size:13px}code{background:#eee;padding:0 4px;border-radius:4px}"
          ".row{display:flex;gap:8px;align-items:center;margin:6px 0}.row label{width:28px}"
          ".row input[type=range]{flex:1}.row input[type=number]{width:70px}"
          ".pred{font-size:18px;font-weight:700;margin:10px 0}canvas{width:100%;border:1px solid #ccc;border-radius:6px}"
          ".nb{display:inline-block;min-width:118px;background:#eee;border-radius:6px;margin:2px 4px 2px 0;"
          "position:relative;overflow:hidden;font-size:11px}.nb i{display:block;height:14px;background:#1f77b4}"
          ".nb b{position:absolute;left:4px;top:0;font-weight:600}.lyr{margin:4px 0}</style></head><body>\n";
    os << "<h1>" << htmlEscape(title_) << "</h1>\n";
    os << "<p class=\"muted\">Minimal report — every CLI visual in one page: loss + accuracy curves, "
          "decision map, network graph, weights, live neuron panel.</p>\n";
    os << svgChart(losses, colors, true, "Training loss", "loss");
    if (!accs.empty()) os << svgChart(accs, colors, false, "Accuracy", "accuracy");
    os << buildProbeSection();
    for (auto& pr : pres_)
        os << "<div class=\"card\"><h2>" << htmlEscape(pr.first) << "</h2>\n<pre>"
           << htmlEscape(pr.second) << "</pre></div>\n";
    for (auto& s : sections_)
        if (s) os << s->toHtml();
    os << "</body></html>\n";
    return os.str();
}

void HtmlReport::save(const std::string& path) {
    std::string html = render();
    std::ofstream f(path);
    if (!f) throw std::runtime_error("HtmlReport::save: cannot open " + path);
    f << html;
}

LiveConsole::LiveConsole(NeuralNetwork& net, Vector probe, std::string title,
                         int totalEpochs, int stride)
    : net_(net), probe_(std::move(probe)), title_(std::move(title)),
      total_(totalEpochs), stride_(stride <= 0 ? 1 : stride) {}

void LiveConsole::onEpoch(int epoch, const TrainingHistory& hist) {
    if (epoch == 1 || epoch == total_ || epoch % stride_ == 0) draw(epoch, hist);
}

namespace {
// Bipolar firing bar: '-' baseline, '|' center, '#' from center to value.
std::string fireBar(double v, double lo, double hi, int w) {
    std::string s(std::size_t(w), '-');
    int c = (w - 1) / 2;
    s[std::size_t(c)] = '|';
    double f = (v - lo) / (hi - lo);
    int p = int(std::lround(f * double(w - 1)));
    p = std::max(0, std::min(w - 1, p));
    for (int i = std::min(c, p); i <= std::max(c, p); ++i)
        if (i != c) s[std::size_t(i)] = '#';
    return s;
}
} // namespace

void LiveConsole::draw(int epoch, const TrainingHistory& hist) {
    net_.predict(probe_); // refresh per-neuron cached outputs for the panel
    std::ostringstream os;
    os << "\x1b[2J\x1b[H"; // clear screen + home (ANSI terminals)
    os << title_ << "   [epoch " << epoch << "/" << total_ << "]";
    if (!hist.trainLoss.empty())
        os << "   loss=" << hist.trainLoss.back() << " acc=" << hist.trainAcc.back();
    os << "\n";
    os << linePlot(hist.trainLoss, 56, 9, true, false, "loss vs epoch (live)");
    os << linePlot(hist.trainAcc, 56, 7, false, true, "accuracy vs epoch (live)");
    os << "neurons @ probe [";
    for (std::size_t i = 0; i < probe_.size(); ++i) os << (i ? "," : "") << probe_[i];
    os << "]  (- = off, # = firing)\n";
    std::size_t li = 0;
    for (auto& layer : net_.layers()) {
        std::string act = layer.neurons()[0].activation().name();
        double lo = -1.0, hi = 1.0;
        if (act == "sigmoid") { lo = 0.0; hi = 1.0; }
        else if (act == "relu" || act == "leaky_relu" || act == "leakyrelu") { lo = 0.0; hi = 1.0; }
        else if (act == "swish") { lo = -0.3; hi = 1.0; }
        os << " L" << li++ << " " << act << "\n";
        std::size_t ji = 0;
        for (auto& n : layer.neurons()) {
            double v = std::max(lo, std::min(hi, n.lastOutput()));
            char buf[16];
            std::snprintf(buf, sizeof buf, "%.2f", n.lastOutput());
            os << "   N" << ji++ << " [" << fireBar(v, lo, hi, 21) << "] " << buf << "\n";
        }
    }
    std::cout << os.str() << std::flush;
}

} // namespace miniann
