#pragma once
// Interactive 3D neural-network viewer (QOpenGLWidget, moc-free: virtual
// overrides + lambdas only, like the rest of the GUI — no Q_OBJECT, no moc
// step). Consumes NetSnapshot3D (same data as the 2D view's timeline), never
// touches training. Neurons = camera-facing billboard quads, connections =
// GL_LINES. Minimal GLSL (attribute/varying subset valid on desktop GL 2.1
// and ES 1.00 alike).
#include <QElapsedTimer>
#include <QKeyEvent>
#include <QList>
#include <QMatrix4x4>
#include <QMouseEvent>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>
#include <QPoint>
#include <QPointer>
#include <QString>
#include <QTimer>
#include <QVector3D>
#include <QWheelEvent>
#include <cstddef>
#include <functional>
#include <map>
#include <tuple>
#include <vector>

#include "camera3d.hpp"
#include "edge_budget.hpp"
#include "layout3d.hpp"
#include "snapshot3d.hpp"

namespace net3d {

class Network3DWidget : public QOpenGLWidget, protected QOpenGLFunctions {
public:
    explicit Network3DWidget(QWidget* parent = nullptr);
    ~Network3DWidget() override;

    // Live snapshot: same content the 2D view shows (weights, probe, forward
    // phase, LIVE/FINAL, hot update edges). Column 0 = input, 1..L = layers.
    // backPhase/backIntensity drive the backward sweep glow (amber, scaled by
    // the real loss drop); -1 disables it.
    void setSnapshot(const NetSnapshot3D& st, int fwdPhase, bool liveMode,
                     std::vector<std::tuple<int, int, int>> hot = {},
                     const QString& tag = QString(), int backPhase = -1, float backIntensity = 0.0f);
    // Focus layer: -1 = all layers; otherwise only that column (and its
    // adjacent connections) stays fully visible, the rest dims.
    void setFocusLayer(int column);
    // Skeleton before training: layout only, neutral colors, no connections.
    void setArchSizes(const std::vector<std::size_t>& sizes);
    void setWeightThreshold(float t);
    // Selection: column 0 = input pseudo-layer, else 0-based weight layer.
    void setSelected(int column, int neuron);
    void clearSelection();
    void resetView();

    // Human-readable info for the currently selected neuron (side panel).
    QString selectedInfo() const;
    // Enlarged live popups (same pattern as the 2D views): every state change
    // is pushed out; the popup keeps its own camera. QPointer auto-nulls.
    void copyFrom(const Network3DWidget& o);
    void pushMirrors();
    // Edge statistics for the "Showing X / Y connections" readout.
    std::size_t visibleEdges() const { return m_visEdges; }
    std::size_t totalEdges() const { return m_totalEdges; }
    // Fired on click-select (moc-free observer instead of signals).
    std::function<void(int column, int neuron)> onNeuronSelected;

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;
    void mouseDoubleClickEvent(QMouseEvent* e) override;
    void wheelEvent(QWheelEvent* e) override;
    void keyPressEvent(QKeyEvent* e) override;

private:
    bool pickNeuron(const QPoint& pos, int& columnOut, int& neuronOut) const;
    void rebuildBuffers();
    void ensureLayout();
    std::string archFingerprint() const;
    void startCamAnim(const Camera3D& to, int durationMs = 500);
    void stepCamAnim();
    void focusNeuron(int column, int neuron);

    QOpenGLShaderProgram* m_dotProg = nullptr;
    QOpenGLShaderProgram* m_lineProg = nullptr;
    unsigned int m_dotVbo = 0;
    unsigned int m_lineVbo = 0;
    unsigned int m_gridVbo = 0;
    bool m_glReady = false;
    // Faint floor grid (spatial reference) + per-column names + title tag.
    std::vector<float> m_gridVerts;
    std::string m_gridFp;
    std::vector<QString> m_colNames;
    QString m_tag;

    Camera3D m_cam;
    bool m_userMovedCam = false;

    NetSnapshot3D m_snap;
    bool m_hasSnap = false;
    std::vector<std::size_t> m_archSizes; // preview skeleton
    std::vector<LayerLayout3D> m_layout;
    std::string m_layoutFp;
    // Activations per column (col 0 = probe input), recomputed on snapshot.
    std::vector<std::vector<double>> m_fire;

    int m_fwdPhase = -1;
    int m_backPhase = -1;
    float m_backIntensity = 0.0f;
    bool m_liveMode = true;
    float m_weightThreshold = 0.05f;
    std::vector<std::tuple<int, int, int>> m_hot;
    int m_focusCol = -1;
    int m_selCol = -1;
    int m_selIdx = -1;
    // Weight-update flashes: edge -> birth tick; alpha decays with age.
    std::map<std::tuple<int, int, int>, int> m_flash;
    int m_tick = 0;
    std::size_t m_visEdges = 0;
    std::size_t m_totalEdges = 0;
    // Smooth camera animation (reset / focus / refit), moc-free QTimer.
    QTimer m_animTimer;
    QElapsedTimer m_animClock;
    Camera3D m_animFrom;
    Camera3D m_animTo;
    int m_animDurMs = 500;
    bool m_animating = false;

    std::size_t m_dotVerts = 0;
    std::size_t m_lineVerts = 0;
    bool m_dirty = true;

    // Picking state from the last paint.
    mutable QMatrix4x4 m_lastProj;
    mutable QMatrix4x4 m_lastView;
    mutable int m_lastW = 1;
    mutable int m_lastH = 1;

    // Mouse gesture state.
    QPoint m_pressPos;
    bool m_orbiting = false;
    bool m_panning = false;
    bool m_moved = false;

    QList<QPointer<Network3DWidget>> m_mirrors;
};

} // namespace net3d
