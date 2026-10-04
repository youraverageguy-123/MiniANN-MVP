#include "network3d_widget.hpp"

#include <QDialog>
#include <QPainter>
#include <QToolTip>
#include <QVBoxLayout>
#include <QVector4D>
#include <algorithm>
#include <cmath>

#include "miniann/activation.hpp"

namespace net3d {
namespace {

// GLSL subset valid on desktop GL 2.1 and ES 1.00 (no precision qualifiers,
// attribute/varying/gl_FragColor only).
const char* kDotVert = R"GLSL(
attribute vec3 aCenter;
attribute vec2 aCorner;
attribute float aSize;
attribute vec3 aColor;
uniform mat4 uView;
uniform mat4 uProj;
varying vec3 vColor;
varying vec2 vP;
void main() {
    vec4 c = uView * vec4(aCenter, 1.0);
    c.xy += aCorner * aSize;
    vColor = aColor;
    vP = aCorner;
    gl_Position = uProj * c;
}
)GLSL";

const char* kDotFrag = R"GLSL(
varying vec3 vColor;
varying vec2 vP;
void main() {
    float r = length(vP);
    if (r > 1.0) discard;
    float core = smoothstep(1.0, 0.15, r);
    vec3 col = vColor * (0.30 + 0.70 * core);
    col += vec3(1.0) * pow(core, 10.0) * 0.30;
    gl_FragColor = vec4(col, 1.0);
}
)GLSL";

const char* kLineVert = R"GLSL(
attribute vec3 aPos;
attribute vec3 aColor;
attribute float aAlpha;
uniform mat4 uView;
uniform mat4 uProj;
varying vec3 vColor;
varying float vA;
void main() {
    vColor = aColor;
    vA = aAlpha;
    gl_Position = uProj * uView * vec4(aPos, 1.0);
}
)GLSL";

const char* kLineFrag = R"GLSL(
varying vec3 vColor;
varying float vA;
void main() {
    gl_FragColor = vec4(vColor, vA);
}
)GLSL";

const std::size_t kMaxDrawLines = 20000;

double activateName(const std::string& actName, double z) {
    try {
        return miniann::ActivationFactory::create(actName)->activate(z);
    } catch (...) {
        return z;
    }
}

QVector3D dimBase() { return QVector3D(0.106f, 0.118f, 0.169f); }
QVector3D brightBlue() { return QVector3D(0.35f, 0.66f, 1.0f); }

struct EdgeCand {
    int l, a, b;
    double w;
};

bool edgeIsHot(const std::vector<std::tuple<int, int, int>>& hot, int l, int a, int b) {
    for (const auto& t : hot)
        if (std::get<0>(t) == l && std::get<1>(t) == a && std::get<2>(t) == b) return true;
    return false;
}

} // namespace

Network3DWidget::Network3DWidget(QWidget* parent) : QOpenGLWidget(parent) {
    setMouseTracking(true);
    setFocusPolicy(Qt::ClickFocus);
    setToolTip(QStringLiteral("Left-drag orbit · wheel zoom · middle/right-drag pan · click neuron · double-click focus/fullscreen · R reset"));
    m_animTimer.setInterval(30);
    connect(&m_animTimer, &QTimer::timeout, this, [this] { stepCamAnim(); });
}

Network3DWidget::~Network3DWidget() {
    // GL resources are owned by the context (widget lives for the app
    // lifetime); nothing to free without a current context here.
}

void Network3DWidget::setSnapshot(const NetSnapshot3D& st, int fwdPhase, bool liveMode,
                                  std::vector<std::tuple<int, int, int>> hot, const QString& tag,
                                  int backPhase, float backIntensity) {
    m_snap = st;
    m_hasSnap = true;
    m_fwdPhase = fwdPhase;
    m_backPhase = backPhase;
    m_backIntensity = std::max(0.0f, std::min(1.0f, backIntensity));
    m_liveMode = liveMode;
    m_hot = std::move(hot);
    if (!tag.isEmpty()) m_tag = tag;
    // Weight-update flash bookkeeping: new hot edges are born now, stale
    // entries age out so flashes fade and never leak.
    ++m_tick;
    for (const auto& t : m_hot)
        m_flash.try_emplace(t, m_tick);
    for (auto it = m_flash.begin(); it != m_flash.end();) {
        if (m_tick - it->second > 90) it = m_flash.erase(it);
        else ++it;
    }
    m_totalEdges = 0;
    for (std::size_t l = 0; l + 1 < m_snap.weights.size() + 1 && l < m_snap.weights.size(); ++l)
        for (const auto& wrow : m_snap.weights[l]) m_totalEdges += wrow.size();
    ensureLayout();
    pushMirrors();
    // Recompute activations for the probe input.
    m_fire.clear();
    if (!m_snap.probe.empty() && !m_snap.weights.empty()) {
        m_fire.push_back(m_snap.probe);
        miniann::Vector cur = m_snap.probe;
        for (std::size_t l = 0; l < m_snap.weights.size(); ++l) {
            const std::size_t n = (l < m_snap.biases.size()) ? m_snap.biases[l].size() : 0;
            miniann::Vector nxt(n, 0.0);
            const std::string an = (l < m_snap.acts.size()) ? m_snap.acts[l] : "linear";
            for (std::size_t j = 0; j < n; ++j) {
                double z = (j < m_snap.biases[l].size()) ? m_snap.biases[l][j] : 0.0;
                if (j < m_snap.weights[l].size()) {
                    const auto& w = m_snap.weights[l][j];
                    for (std::size_t i = 0; i < cur.size() && i < w.size(); ++i) z += cur[i] * w[i];
                }
                nxt[j] = activateName(an, z);
            }
            m_fire.push_back(nxt);
            cur = nxt;
        }
    }
    m_dirty = true;
    update();
}

void Network3DWidget::setArchSizes(const std::vector<std::size_t>& sizes) {
    m_archSizes = sizes;
    m_hasSnap = false;
    m_fire.clear();
    m_selCol = -1;
    m_selIdx = -1;
    m_tag = QStringLiteral("PREVIEW — edit config, then START TRAINING");
    ensureLayout();
    m_dirty = true;
    update();
    pushMirrors();
}

void Network3DWidget::setWeightThreshold(float t) {
    t = std::max(0.0f, t);
    if (t != m_weightThreshold) {
        m_weightThreshold = t;
        m_dirty = true;
        update();
    }
}

void Network3DWidget::setSelected(int column, int neuron) {
    m_selCol = column;
    m_selIdx = neuron;
    m_dirty = true;
    update();
    pushMirrors();
}

void Network3DWidget::clearSelection() {
    setSelected(-1, -1);
}

void Network3DWidget::setFocusLayer(int column) {
    if (column != m_focusCol) {
        m_focusCol = column;
        m_dirty = true;
        update();
        pushMirrors();
    }
}

void Network3DWidget::resetView() {
    Camera3D d;
    d.reset();
    d.fitExtent(layoutExtent(m_layout));
    m_userMovedCam = false;
    startCamAnim(d);
}

void Network3DWidget::startCamAnim(const Camera3D& to, int durationMs) {
    m_animFrom = m_cam;
    m_animTo = to;
    m_animDurMs = std::max(50, durationMs);
    m_animClock.start();
    m_animating = true;
    if (!m_animTimer.isActive()) m_animTimer.start();
}

void Network3DWidget::stepCamAnim() {
    if (!m_animating) {
        m_animTimer.stop();
        return;
    }
    const float t = (float)m_animClock.elapsed() / (float)m_animDurMs;
    if (t >= 1.0f) {
        m_cam = m_animTo;
        m_animating = false;
        m_animTimer.stop();
    } else {
        m_cam = lerpCam(m_animFrom, m_animTo, t);
    }
    update();
}

void Network3DWidget::focusNeuron(int column, int neuron) {
    if (column < 0 || (std::size_t)column >= m_layout.size()) return;
    if (neuron < 0 || (std::size_t)neuron >= m_layout[(std::size_t)column].positions.size()) return;
    Camera3D to = m_cam;
    to.target = m_layout[(std::size_t)column].positions[(std::size_t)neuron];
    to.distance = std::clamp(6.0f, to.minDistance, to.maxDistance);
    m_userMovedCam = true;
    startCamAnim(to);
}

void Network3DWidget::copyFrom(const Network3DWidget& o) {
    m_snap = o.m_snap;
    m_hasSnap = o.m_hasSnap;
    m_archSizes = o.m_archSizes;
    m_layout = o.m_layout;
    m_layoutFp = o.m_layoutFp;
    m_fire = o.m_fire;
    m_gridVerts = o.m_gridVerts;
    m_gridFp = o.m_gridFp;
    m_colNames = o.m_colNames;
    m_tag = o.m_tag;
    m_fwdPhase = o.m_fwdPhase;
    m_backPhase = o.m_backPhase;
    m_backIntensity = o.m_backIntensity;
    m_liveMode = o.m_liveMode;
    m_weightThreshold = o.m_weightThreshold;
    m_hot = o.m_hot;
    m_focusCol = o.m_focusCol;
    m_selCol = o.m_selCol;
    m_selIdx = o.m_selIdx;
    m_flash = o.m_flash;
    m_tick = o.m_tick;
    m_visEdges = o.m_visEdges;
    m_totalEdges = o.m_totalEdges;
    // Camera deliberately NOT copied: the popup keeps its own viewpoint.
    m_dirty = true;
    update();
}

void Network3DWidget::pushMirrors() {
    for (int i = m_mirrors.size() - 1; i >= 0; --i) {
        if (m_mirrors[i].isNull()) {
            m_mirrors.removeAt(i);
        } else {
            m_mirrors[i]->copyFrom(*this);
            m_mirrors[i]->pushMirrors();
        }
    }
}

std::string Network3DWidget::archFingerprint() const {
    std::string fp;
    if (m_hasSnap) {
        fp += std::to_string(m_snap.inDim) + "|";
        for (const auto& b : m_snap.biases) fp += std::to_string(b.size()) + ",";
    } else {
        fp += "arch|";
        for (auto s : m_archSizes) fp += std::to_string(s) + ",";
    }
    return fp;
}

namespace {

QString shortAct3D(const std::string& n) {
    if (n == "sigmoid") return QStringLiteral("Sig");
    if (n == "tanh") return QStringLiteral("Tanh");
    if (n == "relu") return QStringLiteral("ReLU");
    if (n == "leaky_relu") return QStringLiteral("L-ReLU");
    if (n == "swish") return QStringLiteral("Swish");
    if (n == "linear") return QStringLiteral("Lin");
    return QStringLiteral("?");
}

} // namespace

void Network3DWidget::ensureLayout() {
    const std::string fp = archFingerprint();
    if (fp == m_layoutFp && !m_layout.empty()) return;
    m_layoutFp = fp;
    std::vector<std::size_t> sizes;
    std::vector<std::string> acts;
    if (m_hasSnap) {
        sizes.push_back(m_snap.inDim);
        for (const auto& b : m_snap.biases) sizes.push_back(b.size());
        acts = m_snap.acts;
    } else {
        sizes = m_archSizes;
    }
    std::size_t maxN = 0;
    for (auto s : sizes)
        if (s > maxN) maxN = s;
    const float ns = adaptiveNeuronSpacing(maxN);
    m_layout = buildLayout3D(sizes, adaptiveLayerSpacing(ns, sizes.size()), ns);
    // Per-column names for the 2D overlay (Input / H1·Tanh / Output·Sig).
    m_colNames.clear();
    for (std::size_t c = 0; c < m_layout.size(); ++c) {
        if (c == 0) {
            m_colNames.push_back(QStringLiteral("Input"));
        } else {
            const std::size_t li = c - 1;
            const QString an = (li < acts.size()) ? shortAct3D(acts[li]) : QStringLiteral("?");
            const bool last = (c + 1 == m_layout.size());
            m_colNames.push_back((last ? QStringLiteral("Output·") : QStringLiteral("H%1·").arg(c)) + an);
        }
    }
    // Faint floor grid under the lowest neurons (orientation reference).
    m_gridVerts.clear();
    {
        float loX = 0, hiX = 0, loY = 0, hiY = 0, maxZ = 0;
        bool any = false;
        for (const auto& L : m_layout)
            for (const auto& p : L.positions) {
                if (!any) {
                    loX = hiX = p.x();
                    loY = hiY = p.y();
                    maxZ = std::abs(p.z());
                    any = true;
                } else {
                    if (p.x() < loX) loX = p.x();
                    if (p.x() > hiX) hiX = p.x();
                    if (p.y() < loY) loY = p.y();
                    if (p.y() > hiY) hiY = p.y();
                    const float az = std::abs(p.z());
                    if (az > maxZ) maxZ = az;
                }
            }
        if (any) {
            const float y = loY - 1.2f;
            const float x0 = loX - 2.0f, x1 = hiX + 2.0f;
            const float z0 = -(maxZ + 2.0f), z1 = maxZ + 2.0f;
            const QVector3D gc(0.16f, 0.18f, 0.24f);
            auto gridLine = [&](QVector3D a, QVector3D b, const QVector3D& col, float alpha) {
                for (const QVector3D* p : {&a, &b}) {
                    m_gridVerts.push_back(p->x());
                    m_gridVerts.push_back(p->y());
                    m_gridVerts.push_back(p->z());
                    m_gridVerts.push_back(col.x());
                    m_gridVerts.push_back(col.y());
                    m_gridVerts.push_back(col.z());
                    m_gridVerts.push_back(alpha);
                }
            };
            for (float gx = std::floor(x0 / 2.0f) * 2.0f; gx <= x1; gx += 2.0f)
                gridLine(QVector3D(gx, y, z0), QVector3D(gx, y, z1), gc, 1.0f);
            for (float gz = std::floor(z0 / 2.0f) * 2.0f; gz <= z1; gz += 2.0f)
                gridLine(QVector3D(x0, y, gz), QVector3D(x1, y, gz), gc, 1.0f);
            // Faint per-layer guide rectangles (spatial context, dimmer than
            // the network itself).
            const QVector3D guideC(gc.x() * 0.75f, gc.y() * 0.75f, gc.z() * 0.75f);
            for (const auto& L : m_layout) {
                if (L.positions.empty()) continue;
                float ly0 = L.positions[0].y(), ly1 = ly0, lz0 = L.positions[0].z(), lz1 = lz0;
                for (const auto& q : L.positions) {
                    if (q.y() < ly0) ly0 = q.y();
                    if (q.y() > ly1) ly1 = q.y();
                    if (q.z() < lz0) lz0 = q.z();
                    if (q.z() > lz1) lz1 = q.z();
                }
                const float gx = L.positions[0].x();
                const float m = 0.7f;
                const QVector3D c00(gx, ly0 - m, lz0 - m), c01(gx, ly0 - m, lz1 + m);
                const QVector3D c10(gx, ly1 + m, lz0 - m), c11(gx, ly1 + m, lz1 + m);
                gridLine(c00, c01, guideC, 0.8f);
                gridLine(c01, c11, guideC, 0.8f);
                gridLine(c11, c10, guideC, 0.8f);
                gridLine(c10, c00, guideC, 0.8f);
            }
        }
    }
    m_gridFp = fp;
    if (!m_userMovedCam) {
        // Glide (don't jump) to the fitted pose on architecture changes.
        Camera3D d;
        d.reset();
        d.fitExtent(layoutExtent(m_layout));
        startCamAnim(d, 600);
    }
    // Drop selection if out of range.
    if (m_selCol >= 0) {
        if ((std::size_t)m_selCol >= m_layout.size() ||
            m_selIdx < 0 || (std::size_t)m_selIdx >= m_layout[(std::size_t)m_selCol].positions.size()) {
            m_selCol = -1;
            m_selIdx = -1;
        }
    }
    m_dirty = true;
}

void Network3DWidget::initializeGL() {
    initializeOpenGLFunctions();
    if (!m_dotProg) {
        m_dotProg = new QOpenGLShaderProgram(this);
        m_dotProg->addShaderFromSourceCode(QOpenGLShader::Vertex, kDotVert);
        m_dotProg->addShaderFromSourceCode(QOpenGLShader::Fragment, kDotFrag);
        m_dotProg->bindAttributeLocation("aCenter", 0);
        m_dotProg->bindAttributeLocation("aCorner", 1);
        m_dotProg->bindAttributeLocation("aSize", 2);
        m_dotProg->bindAttributeLocation("aColor", 3);
        if (!m_dotProg->link()) m_dotProg = nullptr;
    }
    if (!m_lineProg) {
        m_lineProg = new QOpenGLShaderProgram(this);
        m_lineProg->addShaderFromSourceCode(QOpenGLShader::Vertex, kLineVert);
        m_lineProg->addShaderFromSourceCode(QOpenGLShader::Fragment, kLineFrag);
        m_lineProg->bindAttributeLocation("aPos", 0);
        m_lineProg->bindAttributeLocation("aColor", 1);
        m_lineProg->bindAttributeLocation("aAlpha", 2);
        if (!m_lineProg->link()) m_lineProg = nullptr;
    }
    if (!m_dotVbo) glGenBuffers(1, &m_dotVbo);
    if (!m_lineVbo) glGenBuffers(1, &m_lineVbo);
    if (!m_gridVbo) glGenBuffers(1, &m_gridVbo);
    m_glReady = (m_dotProg != nullptr && m_lineProg != nullptr && m_dotVbo != 0 && m_lineVbo != 0 &&
                 m_gridVbo != 0);
}

void Network3DWidget::resizeGL(int w, int h) {
    if (w < 1) w = 1;
    if (h < 1) h = 1;
    const float dpr = devicePixelRatio();
    glViewport(0, 0, int(w * dpr), int(h * dpr));
}

namespace {

void pushQuad(std::vector<float>& v, const QVector3D& c, float size, const QVector3D& col) {
    const float cx[6] = {-1, 1, 1, -1, 1, -1};
    const float cy[6] = {-1, -1, 1, -1, 1, 1};
    for (int k = 0; k < 6; ++k) {
        v.push_back(c.x());
        v.push_back(c.y());
        v.push_back(c.z());
        v.push_back(cx[k]);
        v.push_back(cy[k]);
        v.push_back(size);
        v.push_back(col.x());
        v.push_back(col.y());
        v.push_back(col.z());
    }
}

float clamp01(double v) {
    if (v < 0.0) return 0.0f;
    if (v > 1.0) return 1.0f;
    return (float)v;
}

} // namespace

void Network3DWidget::rebuildBuffers() {
    m_dirty = false;
    m_dotVerts = 0;
    m_lineVerts = 0;
    if (m_layout.empty()) return;

    // --- firing values per column ---
    const std::size_t nCols = m_layout.size();
    auto fireAt = [&](std::size_t col, std::size_t i) -> float {
        if (col < m_fire.size() && i < m_fire[col].size()) {
            double f = m_fire[col][i];
            if (m_fire[col].size() > 1) return clamp01(f);
            return clamp01(std::abs(f));
        }
        return -1.0f;
    };

    // --- connections (weight layers connect col l -> col l+1) ---
    // Opacity hierarchy: weak/medium 15-25%, medium/strong 35-60%, strong
    // 80-100%, selected 100%, flashes decay from 100%.
    auto tierAlpha = [&](double absW) {
        const float s = std::min(1.0f, (float)(absW / std::max(1.0, (double)m_weightThreshold * 6.0)));
        if (s < 1.0f / 3.0f) return 0.15f + 0.10f * (s * 3.0f);
        if (s < 2.0f / 3.0f) return 0.35f + 0.25f * ((s - 1.0f / 3.0f) * 3.0f);
        return 0.80f + 0.20f * ((s - 2.0f / 3.0f) * 3.0f);
    };
    std::vector<EdgeW> cands;
    if (m_hasSnap) {
        for (std::size_t l = 0; l + 1 < nCols && l < m_snap.weights.size(); ++l) {
            const auto& W = m_snap.weights[l];
            for (std::size_t b = 0; b < W.size() && b < m_layout[l + 1].positions.size(); ++b)
                for (std::size_t a = 0; a < W[b].size() && a < m_layout[l].positions.size(); ++a) {
                    const double w = W[b][a];
                    const bool hot = edgeIsHot(m_hot, (int)l, (int)a, (int)b);
                    const bool sel = (m_selCol >= 0) &&
                        (((int)l + 1 == m_selCol && (int)b == m_selIdx) ||
                         ((int)l == m_selCol && (int)a == m_selIdx));
                    if (std::abs(w) >= m_weightThreshold || hot || sel) {
                        EdgeW e;
                        e.l = (int)l;
                        e.a = (int)a;
                        e.b = (int)b;
                        e.w = w;
                        e.forced = hot || sel;
                        cands.push_back(e);
                    }
                }
        }
        cands = applyEdgeBudget(std::move(cands), kMaxDrawLines);
    }
    m_visEdges = cands.size();

    const QVector3D kYellow(0.98f, 0.80f, 0.08f);
    const QVector3D kAmber(1.0f, 0.62f, 0.15f);
    const bool focusOn = (m_focusCol >= 0);
    std::vector<float> lines;
    lines.reserve(cands.size() * 14);
    const bool dimOthers = (m_selCol >= 0);
    for (const auto& e : cands) {
        const QVector3D& pa = m_layout[(std::size_t)e.l].positions[(std::size_t)e.a];
        const QVector3D& pb = m_layout[(std::size_t)e.l + 1].positions[(std::size_t)e.b];
        const double absW = std::abs(e.w);
        float bright = 0.25f + 0.75f * std::min(1.0f, (float)absW);
        QVector3D col = (e.w >= 0) ? QVector3D(0.35f, 0.66f, 0.90f) : QVector3D(0.96f, 0.62f, 0.04f);
        col *= bright;
        float alpha = tierAlpha(absW);
        const bool hot = edgeIsHot(m_hot, e.l, e.a, e.b);
        const bool sel = (m_selCol >= 0) && (((e.l + 1 == m_selCol && e.b == m_selIdx) ||
                                             (e.l == m_selCol && e.a == m_selIdx)));
        const bool fwd = m_liveMode && e.l == m_fwdPhase;
        const bool bwd = m_liveMode && e.l == m_backPhase;
        const bool incidentFocus = !focusOn || e.l == m_focusCol - 1 || e.l == m_focusCol;
        if (hot) {
            // Weight-update flash: yellow at birth, fading to base color.
            float flash = 1.0f;
            auto it = m_flash.find(std::make_tuple(e.l, e.a, e.b));
            if (it != m_flash.end()) flash = std::exp(-(float)(m_tick - it->second) / 6.0f);
            const float k = 0.25f + 0.75f * flash;
            col = col * (1.0f - k) + kYellow * k;
            alpha = std::max(alpha, 0.25f + 0.75f * flash);
        } else if (fwd) {
            col = col * 0.5f + QVector3D(0.5f, 0.5f, 0.5f);
            alpha = 0.95f;
        } else if (bwd) {
            const float k = m_backIntensity;
            col = col * (1.0f - 0.6f * k) + kAmber * (0.6f * k);
            alpha = std::max(alpha, 0.5f + 0.45f * k);
        } else if (sel) {
            col = col * 0.6f + QVector3D(0.4f, 0.4f, 0.4f);
            alpha = 1.0f;
        } else if (!m_liveMode) {
            col *= 0.85f;
        }
        if (dimOthers && !sel) {
            col *= 0.35f;
            alpha *= 0.5f;
        }
        if (focusOn && !incidentFocus) {
            col *= 0.35f;
            alpha *= 0.5f;
        }
        for (const QVector3D* p : {&pa, &pb}) {
            lines.push_back(p->x());
            lines.push_back(p->y());
            lines.push_back(p->z());
            lines.push_back(col.x());
            lines.push_back(col.y());
            lines.push_back(col.z());
            lines.push_back(alpha);
        }
    }

    // --- neurons: radius = base*(0.7+0.8*a), brightness = glow(a); tanh
    // sign encoded blue (pos) / orange (neg), near-zero dim.
    std::vector<float> dots;
    const QVector3D camPos = m_cam.position();
    const QVector3D kNeg(0.96f, 0.55f, 0.10f);
    const QVector3D kZero(0.23f, 0.26f, 0.33f);
    for (std::size_t c = 0; c < nCols; ++c) {
        const bool colActive = m_liveMode && ((int)c == m_fwdPhase + 1 || (int)c - 1 == m_backPhase);
        const bool colDimFocus = focusOn && (int)c != m_focusCol;
        for (std::size_t i = 0; i < m_layout[c].positions.size(); ++i) {
            const QVector3D& p = m_layout[c].positions[i];
            const float f = fireAt(c, i);
            const bool isSel = ((int)c == m_selCol && (int)i == m_selIdx);
            QVector3D col;
            float size;
            if (f < 0.0f) {
                col = QVector3D(0.20f, 0.23f, 0.30f);
                size = 0.30f * 0.7f;
            } else {
                double raw = 0.0;
                if (c < m_fire.size() && i < m_fire[c].size()) raw = m_fire[c][i];
                const float mag = clamp01(std::abs(raw));
                if (c > 0 && raw < -0.05) {
                    const float k = std::min(1.0f, (float)(-raw));
                    col = kZero * (1.0f - k) + kNeg * k;
                } else if (c > 0 && std::abs(raw) <= 0.05) {
                    col = kZero;
                } else {
                    col = dimBase() + (brightBlue() - dimBase()) * mag;
                }
                size = 0.30f * (0.7f + 0.8f * mag);
                if (size > 0.6f) size = 0.6f;
            }
            if ((dimOthers && !isSel) || colDimFocus) col *= 0.45f;
            if (m_focusCol >= 0 && (int)c == m_focusCol && !isSel) size *= 1.1f;
            if (isSel) {
                // White underlay pushed slightly toward the camera.
                QVector3D dir = p - camPos;
                if (dir.lengthSquared() > 1e-8f) dir.normalize();
                pushQuad(dots, p + dir * size * 0.35f, size * 1.9f, QVector3D(0.92f, 0.95f, 1.0f));
                col = col * 0.5f + QVector3D(0.5f, 0.55f, 0.6f);
                if (colActive) size *= 1.15f;
            } else if (colActive) {
                size *= 1.18f;
            }
            pushQuad(dots, p, size, col);
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, m_lineVbo);
    glBufferData(GL_ARRAY_BUFFER, (long long)(lines.size() * sizeof(float)), lines.empty() ? nullptr : lines.data(),
                 GL_DYNAMIC_DRAW);
    m_lineVerts = lines.size() / 7; // 7 floats/vert: pos(3) + color(3) + alpha(1)
    glBindBuffer(GL_ARRAY_BUFFER, m_dotVbo);
    glBufferData(GL_ARRAY_BUFFER, (long long)(dots.size() * sizeof(float)), dots.empty() ? nullptr : dots.data(),
                 GL_DYNAMIC_DRAW);
    m_dotVerts = dots.size() / 9;
    glBindBuffer(GL_ARRAY_BUFFER, m_gridVbo);
    glBufferData(GL_ARRAY_BUFFER, (long long)(m_gridVerts.size() * sizeof(float)),
                 m_gridVerts.empty() ? nullptr : m_gridVerts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Network3DWidget::paintGL() {
    const float dpr = devicePixelRatio();
    glViewport(0, 0, int(width() * dpr), int(height() * dpr));
    glClearColor(0.035f, 0.039f, 0.059f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    if (!m_glReady || (m_layout.empty())) return;
    if (m_dirty) rebuildBuffers();

    QMatrix4x4 view = m_cam.viewMatrix();
    QMatrix4x4 proj;
    const float aspect = height() > 0 ? (float)width() / (float)height() : 1.0f;
    proj.perspective(m_cam.fovDeg, aspect, 0.1f, 1000.0f);
    m_lastProj = proj;
    m_lastView = view;
    m_lastW = width();
    m_lastH = height();

    auto drawLineVbo = [&](unsigned int vbo, std::size_t nVerts) {
        if (nVerts == 0) return;
        m_lineProg->bind();
        m_lineProg->setUniformValue("uView", view);
        m_lineProg->setUniformValue("uProj", proj);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        m_lineProg->enableAttributeArray(0);
        m_lineProg->setAttributeBuffer(0, GL_FLOAT, 0, 3, 7 * sizeof(float));
        m_lineProg->enableAttributeArray(1);
        m_lineProg->setAttributeBuffer(1, GL_FLOAT, 3 * sizeof(float), 3, 7 * sizeof(float));
        m_lineProg->enableAttributeArray(2);
        m_lineProg->setAttributeBuffer(2, GL_FLOAT, 6 * sizeof(float), 1, 7 * sizeof(float));
        glDrawArrays(GL_LINES, 0, (int)nVerts);
        for (int a = 0; a < 3; ++a) m_lineProg->disableAttributeArray(a);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        m_lineProg->release();
    };
    // Transparent line pass: grid + edges blend over the background, neurons
    // stay opaque on top.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    drawLineVbo(m_gridVbo, m_gridVerts.size() / 7); // floor + guides first
    drawLineVbo(m_lineVbo, m_lineVerts);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    if (m_dotVerts > 0) {
        m_dotProg->bind();
        m_dotProg->setUniformValue("uView", view);
        m_dotProg->setUniformValue("uProj", proj);
        glBindBuffer(GL_ARRAY_BUFFER, m_dotVbo);
        const int stride = 9 * sizeof(float);
        m_dotProg->enableAttributeArray(0);
        m_dotProg->setAttributeBuffer(0, GL_FLOAT, 0, 3, stride);
        m_dotProg->enableAttributeArray(1);
        m_dotProg->setAttributeBuffer(1, GL_FLOAT, 3 * sizeof(float), 2, stride);
        m_dotProg->enableAttributeArray(2);
        m_dotProg->setAttributeBuffer(2, GL_FLOAT, 5 * sizeof(float), 1, stride);
        m_dotProg->enableAttributeArray(3);
        m_dotProg->setAttributeBuffer(3, GL_FLOAT, 6 * sizeof(float), 3, stride);
        glDrawArrays(GL_TRIANGLES, 0, (int)m_dotVerts);
        for (int a = 0; a < 4; ++a) m_dotProg->disableAttributeArray(a);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        m_dotProg->release();
    }

    // 2D overlay: title tag, per-layer names projected from 3D, control hint.
    // This is the spatial reference that keeps fullscreen orbiting readable.
    {
        QPainter p(this);
        QFont f = font();
        f.setPointSize(8);
        p.setFont(f);
        auto toScreen = [&](const QVector3D& w, QPointF& out) -> bool {
            const QMatrix4x4 mvp = m_lastProj * m_lastView;
            const QVector4D v = mvp * QVector4D(w, 1.0f);
            if (v.w() <= 0.0f) return false;
            const QVector3D ndc = v.toVector3DAffine();
            if (ndc.z() < -1.0f || ndc.z() > 1.0f) return false;
            out = QPointF((ndc.x() * 0.5 + 0.5) * m_lastW, (1.0 - (ndc.y() * 0.5 + 0.5)) * m_lastH);
            return true;
        };
        if (!m_tag.isEmpty()) {
            p.setPen(QColor(0x8A, 0x90, 0xA0));
            QFont tf = f;
            tf.setBold(true);
            p.setFont(tf);
            p.drawText(12, 8, m_lastW - 120, 20, Qt::AlignLeft | Qt::AlignVCenter,
                       fontMetrics().elidedText(m_tag, Qt::ElideRight, m_lastW - 120));
            p.setFont(f);
        }
        // LIVE / FINAL badge, top-right: instantly shows whether this is an
        // active training state or the finished model.
        {
            const bool live = m_liveMode;
            const QString txt = live ? QStringLiteral("LIVE") : QStringLiteral("FINAL");
            QFont bf = f;
            bf.setBold(true);
            p.setFont(bf);
            const int bw = 64, bh = 20;
            const QRect pill(m_lastW - bw - 10, 8, bw, bh);
            p.setPen(Qt::NoPen);
            p.setBrush(live ? QColor(0x14, 0x53, 0x2D) : QColor(0x13, 0x4E, 0x4A));
            p.drawRoundedRect(pill, 6, 6);
            p.setPen(live ? QColor(0x22, 0xC5, 0x5E) : QColor(0x2D, 0xD4, 0xBF));
            p.drawText(pill, Qt::AlignCenter, txt);
            p.setFont(f);
        }
        p.setPen(QColor(0xD0, 0xD3, 0xDB));
        for (std::size_t c = 0; c < m_layout.size() && c < m_colNames.size(); ++c) {
            if (m_layout[c].positions.empty()) continue;
            float topY = m_layout[c].positions[0].y();
            for (const auto& q : m_layout[c].positions)
                if (q.y() > topY) topY = q.y();
            QPointF s;
            if (!toScreen(QVector3D(m_layout[c].positions[0].x(), topY + 0.9f, 0.0f), s)) continue;
            const QString t = m_colNames[c];
            const int tw = fontMetrics().horizontalAdvance(t);
            p.drawText(int(s.x()) - tw / 2, int(s.y()) - 8, tw + 4, 16, Qt::AlignLeft | Qt::AlignVCenter, t);
        }
        // Selected-neuron tag pinned above the neuron (persists, unlike hover).
        if (m_selCol >= 0 && m_selIdx >= 0 && (std::size_t)m_selCol < m_layout.size() &&
            (std::size_t)m_selIdx < m_layout[(std::size_t)m_selCol].positions.size()) {
            QPointF s;
            const QVector3D& sp = m_layout[(std::size_t)m_selCol].positions[(std::size_t)m_selIdx];
            double av = 0.0;
            if ((std::size_t)m_selCol < m_fire.size() && (std::size_t)m_selIdx < m_fire[(std::size_t)m_selCol].size())
                av = m_fire[(std::size_t)m_selCol][(std::size_t)m_selIdx];
            QString lbl;
            if (m_selCol == 0) {
                lbl = QString::asprintf("Input[%d]", m_selIdx);
            } else {
                const std::size_t li = (std::size_t)m_selCol - 1;
                const bool isOut = m_hasSnap && li + 1 == m_snap.weights.size();
                lbl = QString::asprintf("%s[%d] %.2f", isOut ? "OUT" : ("H" + std::to_string(m_selCol)).c_str(),
                                        m_selIdx, av);
            }
            // Project manually (same math as the layer labels above).
            const QMatrix4x4 mvp = m_lastProj * m_lastView;
            const QVector4D v = mvp * QVector4D(sp, 1.0f);
            if (v.w() > 0.0f) {
                const QVector3D ndc = v.toVector3DAffine();
                if (ndc.z() >= -1.0f && ndc.z() <= 1.0f) {
                    const float sx = (ndc.x() * 0.5f + 0.5f) * (float)m_lastW;
                    const float sy = (1.0f - (ndc.y() * 0.5f + 0.5f)) * (float)m_lastH;
                    const int tw = fontMetrics().horizontalAdvance(lbl);
                    p.setPen(Qt::NoPen);
                    p.setBrush(QColor(0x0E, 0x11, 0x1A));
                    p.drawRoundedRect(int(sx) - tw / 2 - 6, int(sy) - 34, tw + 12, 18, 4, 4);
                    p.setPen(QColor(0xF1, 0xF5, 0xF9));
                    p.drawText(int(sx) - tw / 2 - 6, int(sy) - 34, tw + 12, 18, Qt::AlignCenter, lbl);
                }
            }
        }
        p.setPen(QColor(0x8A, 0x90, 0xA0));
        p.drawText(12, m_lastH - 24, m_lastW - 24, 16, Qt::AlignLeft | Qt::AlignVCenter,
                   QStringLiteral("Left-drag orbit  |  Wheel zoom  |  Right-drag pan  |  Click neuron  |  Dbl-click empty fullscreen  |  R reset"));
        // Compact legend, bottom-right: activation ramp + weight key.
        {
            const int lw = 168, lh = 76;
            const int lx = m_lastW - lw - 10, ly = m_lastH - lh - 10;
            p.setPen(QPen(QColor(0x2E, 0x34, 0x48), 1));
            p.setBrush(QColor(0x0E, 0x11, 0x1A));
            p.drawRoundedRect(lx, ly, lw, lh, 5, 5);
            p.setPen(QColor(0xD0, 0xD3, 0xDB));
            p.drawText(lx + 8, ly + 4, lw - 16, 14, Qt::AlignLeft | Qt::AlignVCenter,
                       QStringLiteral("ACT low -> high"));
            for (int k = 0; k < 12; ++k) {
                const float t = (float)k / 11.0f;
                const QVector3D c = QVector3D(0.106f, 0.118f, 0.169f) * (1.0f - t) +
                                    QVector3D(0.35f, 0.66f, 1.0f) * t;
                p.setPen(Qt::NoPen);
                p.setBrush(QColor(int(c.x() * 255), int(c.y() * 255), int(c.z() * 255)));
                p.drawRect(lx + 8 + k * 12, ly + 20, 12, 10);
            }
            struct Swatch {
                const char* txt;
                int r, g, b;
            };
            const Swatch sw[3] = {{"positive", 0x5A, 0xA9, 0xE6},
                                  {"negative", 0xF5, 0x9E, 0x0B},
                                  {"changed", 0xFA, 0xCC, 0x15}};
            p.setFont(f);
            for (int k = 0; k < 3; ++k) {
                const int ry = ly + 34 + k * 13;
                p.setPen(Qt::NoPen);
                p.setBrush(QColor(sw[k].r, sw[k].g, sw[k].b));
                p.drawRect(lx + 8, ry, 14, 8);
                p.setPen(QColor(0xD0, 0xD3, 0xDB));
                p.drawText(lx + 26, ry - 3, lw - 34, 14, Qt::AlignLeft | Qt::AlignVCenter,
                           QString::fromLatin1(sw[k].txt));
            }
        }
    }
}

bool Network3DWidget::pickNeuron(const QPoint& pos, int& columnOut, int& neuronOut) const {
    if (m_layout.empty() || m_lastW <= 0 || m_lastH <= 0) return false;
    const QMatrix4x4 mvp = m_lastProj * m_lastView;
    int bestC = -1, bestI = -1;
    float bestD = 16.0f;
    for (std::size_t c = 0; c < m_layout.size(); ++c) {
        for (std::size_t i = 0; i < m_layout[c].positions.size(); ++i) {
            const QVector4D v = mvp * QVector4D(m_layout[c].positions[i], 1.0f);
            if (v.w() <= 0.0f) continue; // behind camera
            const QVector3D ndc = v.toVector3DAffine();
            if (ndc.z() < -1.0f || ndc.z() > 1.0f) continue;
            const float sx = (ndc.x() * 0.5f + 0.5f) * (float)m_lastW;
            const float sy = (1.0f - (ndc.y() * 0.5f + 0.5f)) * (float)m_lastH;
            const float dx = sx - (float)pos.x(), dy = sy - (float)pos.y();
            const float d = std::sqrt(dx * dx + dy * dy);
            if (d < bestD) {
                bestD = d;
                bestC = (int)c;
                bestI = (int)i;
            }
        }
    }
    if (bestC < 0) return false;
    columnOut = bestC;
    neuronOut = bestI;
    return true;
}

QString Network3DWidget::selectedInfo() const {
    if (m_selCol < 0 || m_selIdx < 0) return QStringLiteral("3D: click a neuron to inspect.");
    if ((std::size_t)m_selCol >= m_layout.size()) return QStringLiteral("3D: click a neuron to inspect.");
    if (m_selCol == 0) {
        double v = 0.0;
        if (!m_snap.probe.empty() && (std::size_t)m_selIdx < m_snap.probe.size()) v = m_snap.probe[(std::size_t)m_selIdx];
        return QString::asprintf("3D Input[%d]  value %.4f", m_selIdx, v);
    }
    const std::size_t li = (std::size_t)m_selCol - 1;
    if (!m_hasSnap || li >= m_snap.weights.size() || (std::size_t)m_selIdx >= m_snap.weights[li].size())
        return QString::asprintf("3D H%llu[%d]  (no snapshot yet)", (unsigned long long)m_selCol, m_selIdx);
    const std::string an = (li < m_snap.acts.size()) ? m_snap.acts[li] : "?";
    const double b = (li < m_snap.biases.size() && (std::size_t)m_selIdx < m_snap.biases[li].size())
                         ? m_snap.biases[li][(std::size_t)m_selIdx]
                         : 0.0;
    double act = 0.0;
    if ((std::size_t)m_selCol < m_fire.size() && (std::size_t)m_selIdx < m_fire[(std::size_t)m_selCol].size())
        act = m_fire[(std::size_t)m_selCol][(std::size_t)m_selIdx];
    const bool isOut = (li + 1 == m_snap.weights.size());
    QString s = QString::asprintf("3D %s[%d]  act %.3f  bias %+.3f  (%s)%s", isOut ? "OUT" : ("H" + std::to_string(m_selCol)).c_str(),
                                  m_selIdx, act, b, an.c_str(), isOut ? "" : "");
    // Top incoming / outgoing by |w|.
    const auto& wIn = m_snap.weights[li][(std::size_t)m_selIdx];
    std::vector<std::pair<double, int>> in;
    for (std::size_t k = 0; k < wIn.size(); ++k) in.emplace_back(std::abs(wIn[k]), (int)k);
    std::sort(in.begin(), in.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
    QString inTxt;
    for (std::size_t k = 0; k < in.size() && k < 8; ++k)
        inTxt += QString::asprintf("%sin[%d] %+.2f", k ? " " : "", in[k].second, wIn[(std::size_t)in[k].second]);
    QString outTxt;
    if (!isOut && li + 1 < m_snap.weights.size()) {
        std::vector<std::pair<double, int>> out;
        for (std::size_t j = 0; j < m_snap.weights[li + 1].size(); ++j) {
            const auto& w = m_snap.weights[li + 1][j];
            if ((std::size_t)m_selIdx < w.size()) out.emplace_back(std::abs(w[(std::size_t)m_selIdx]), (int)j);
        }
        std::sort(out.begin(), out.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
        for (std::size_t k = 0; k < out.size() && k < 8; ++k) {
            const double w = m_snap.weights[li + 1][(std::size_t)out[k].second][(std::size_t)m_selIdx];
            outTxt += QString::asprintf("%sout[%d] %+.2f", k ? " " : "", out[k].second, w);
        }
    }
    s += QStringLiteral("  in: ") + (inTxt.isEmpty() ? QStringLiteral("-") : inTxt);
    if (!outTxt.isEmpty()) s += QStringLiteral("  out: ") + outTxt;
    return s;
}

void Network3DWidget::mousePressEvent(QMouseEvent* e) {
    m_pressPos = e->pos();
    m_moved = false;
    // The user takes over: stop any camera glide.
    m_animating = false;
    if (m_animTimer.isActive()) m_animTimer.stop();
    if (e->button() == Qt::LeftButton) m_orbiting = true;
    if (e->button() == Qt::MiddleButton || e->button() == Qt::RightButton) m_panning = true;
}

void Network3DWidget::mouseMoveEvent(QMouseEvent* e) {
    if (m_orbiting && (e->buttons() & Qt::LeftButton)) {
        if ((std::abs(e->pos().x() - m_pressPos.x()) + std::abs(e->pos().y() - m_pressPos.y())) > 3) m_moved = true;
        m_cam.orbit((float)(e->pos().x() - m_pressPos.x()), (float)(e->pos().y() - m_pressPos.y()));
        m_pressPos = e->pos();
        m_userMovedCam = true;
        update();
        return;
    }
    if (m_panning && (e->buttons() & (Qt::MiddleButton | Qt::RightButton))) {
        m_cam.pan((float)(e->pos().x() - m_pressPos.x()), -(float)(e->pos().y() - m_pressPos.y()),
                  (float)std::max(1, height()));
        m_pressPos = e->pos();
        m_userMovedCam = true;
        m_moved = true;
        update();
        return;
    }
    if (e->buttons() == Qt::NoButton) {
        int c = -1, i = -1;
        if (pickNeuron(e->pos(), c, i)) {
            QString tip;
            if (c == 0) {
                tip = QString::asprintf("Input[%d]", i);
            } else {
                tip = QString::asprintf("%s[%d]", ((std::size_t)(c - 1) + 1 == m_snap.weights.size()) ? "OUT"
                                        : ("H" + std::to_string(c)).c_str(),
                                        i);
            }
            QToolTip::showText(e->globalPosition().toPoint(), tip, this);
        } else {
            QToolTip::hideText();
        }
    }
}

void Network3DWidget::mouseDoubleClickEvent(QMouseEvent* e) {
    if (e->button() != Qt::LeftButton) return;
    int c = -1, i = -1;
    if (pickNeuron(e->pos(), c, i)) {
        setSelected(c, i);
        if (onNeuronSelected) onNeuronSelected(c, i);
        focusNeuron(c, i); // select + glide the camera onto it
        return;
    }
    clearSelection();
    if (onNeuronSelected) onNeuronSelected(-1, -1);
    // Empty space: enlarged live view, like the other graphs.
    auto* dlg = new QDialog(window());
    dlg->setWindowTitle(m_tag.isEmpty() ? QStringLiteral("Network 3D (live)") : m_tag + QStringLiteral(" (live)"));
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->resize(900, 650);
    auto* lay = new QVBoxLayout(dlg);
    lay->setContentsMargins(6, 6, 6, 6);
    auto* big = new Network3DWidget;
    big->copyFrom(*this);
    big->m_cam = m_cam; // start from the same viewpoint (independent after)
    m_mirrors.push_back(big);
    lay->addWidget(big);
    dlg->show();
}

void Network3DWidget::mouseReleaseEvent(QMouseEvent* e) {
    const bool wasOrbit = m_orbiting;
    m_orbiting = false;
    m_panning = false;
    if (e->button() == Qt::LeftButton && wasOrbit && !m_moved) {
        int c = -1, i = -1;
        if (pickNeuron(e->pos(), c, i)) {
            setSelected(c, i);
            if (onNeuronSelected) onNeuronSelected(c, i);
        } else {
            clearSelection();
            if (onNeuronSelected) onNeuronSelected(-1, -1);
        }
    }
}

void Network3DWidget::wheelEvent(QWheelEvent* e) {
    const float steps = (float)e->angleDelta().y() / 120.0f;
    if (steps != 0.0f) {
        m_cam.zoomSteps(steps);
        m_userMovedCam = true;
        update();
    }
}

void Network3DWidget::keyPressEvent(QKeyEvent* e) {
    if (e->key() == Qt::Key_R) {
        resetView();
        return;
    }
    QOpenGLWidget::keyPressEvent(e);
}

} // namespace net3d
