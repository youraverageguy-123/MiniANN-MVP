#pragma once
// Orbit camera: the network stays fixed, the camera moves around a target.
// Pure math (QVector3D/QMatrix4x4 only) — no GL context needed, unit-tested
// in tests/test_viz3d_math.cpp. Moc-free.
#include <QMatrix4x4>
#include <QVector3D>
#include <QtMath>
#include <algorithm>

namespace net3d {

struct Camera3D {
    QVector3D target{0.0f, 0.0f, 0.0f};
    float yawDeg = 35.0f;   // orbit around Y
    float pitchDeg = 18.0f; // elevation, clamped to [-89, 89]
    float distance = 18.0f;
    float minDistance = 3.0f;
    float maxDistance = 220.0f;
    float fovDeg = 45.0f;

    QVector3D position() const {
        const float yr = qDegreesToRadians(yawDeg);
        const float pr = qDegreesToRadians(pitchDeg);
        const float cp = std::cos(pr);
        return QVector3D(target.x() + distance * cp * std::sin(yr),
                         target.y() + distance * std::sin(pr),
                         target.z() + distance * cp * std::cos(yr));
    }

    QMatrix4x4 viewMatrix() const {
        QMatrix4x4 v;
        v.lookAt(position(), target, QVector3D(0.0f, 1.0f, 0.0f));
        return v;
    }

    // Left-drag: grab-style orbit, the scene follows the cursor (drag right
    // moves content right, i.e. the camera travels left around the target).
    // Vertical follows the cursor too (drag down tips the top toward you).
    void orbit(float dxPx, float dyPx, float sensitivity = 0.35f) {
        yawDeg -= dxPx * sensitivity;
        pitchDeg = std::clamp(pitchDeg + dyPx * sensitivity, -89.0f, 89.0f);
    }

    // Wheel: steps > 0 zooms in.
    void zoomSteps(float steps, float factor = 0.12f) {
        distance = std::clamp(distance * (1.0f - steps * factor), minDistance, maxDistance);
    }

    // Middle/right-drag: move the target in the camera plane. Scale converts
    // pixels to world units at the target depth for the current fov/distance.
    void pan(float dxPx, float dyPx, float viewportHeightPx) {
        if (viewportHeightPx <= 0.0f) return;
        const float worldPerPx =
            2.0f * distance * std::tan(qDegreesToRadians(fovDeg) * 0.5f) / viewportHeightPx;
        const QVector3D fwd = (target - position()).normalized();
        const QVector3D right = QVector3D::crossProduct(fwd, QVector3D(0.0f, 1.0f, 0.0f)).normalized();
        const QVector3D up = QVector3D::crossProduct(right, fwd).normalized();
        target -= right * dxPx * worldPerPx;
        target += up * dyPx * worldPerPx;
    }

    void reset(float defaultYaw = 35.0f, float defaultPitch = 18.0f, float defaultDist = 18.0f) {
        target = QVector3D(0.0f, 0.0f, 0.0f);
        yawDeg = defaultYaw;
        pitchDeg = defaultPitch;
        distance = std::clamp(defaultDist, minDistance, maxDistance);
    }

    // Fit a scene of the given world extent: distance so it fills ~70% height.
    void fitExtent(float extent) {
        if (extent <= 0.0f) return;
        const float need = (extent * 0.5f) / std::tan(qDegreesToRadians(fovDeg) * 0.5f);
        distance = std::clamp(need * 1.35f, minDistance, maxDistance);
    }
};

// Smoothstep-interpolated camera pose for animated transitions (reset,
// neuron focus, architecture refit). Pure math — unit-tested. Yaw takes the
// shortest path around the circle.
inline Camera3D lerpCam(const Camera3D& a, const Camera3D& b, float t) {
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    const float e = t * t * (3.0f - 2.0f * t);
    Camera3D o = a;
    float dy = b.yawDeg - a.yawDeg;
    while (dy > 180.0f) dy -= 360.0f;
    while (dy < -180.0f) dy += 360.0f;
    o.yawDeg = a.yawDeg + dy * e;
    o.pitchDeg = a.pitchDeg + (b.pitchDeg - a.pitchDeg) * e;
    o.distance = a.distance + (b.distance - a.distance) * e;
    o.target = a.target + (b.target - a.target) * e;
    return o;
}

} // namespace net3d
