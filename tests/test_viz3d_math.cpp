// Camera + layout math for the 3D viewer (pure logic, no GL context).
#include <cassert>
#include <cmath>
#include <iostream>
#include "../demos/network3d/camera3d.hpp"
#include "../demos/network3d/edge_budget.hpp"
#include "../demos/network3d/layout3d.hpp"

using namespace net3d;

static bool near(float a, float b, float eps = 1e-4f) { return std::abs(a - b) < eps; }

int main() {
    // Orbit position: yaw=0, pitch=0 -> +Z axis at distance.
    {
        Camera3D c;
        c.target = QVector3D(0, 0, 0);
        c.yawDeg = 0;
        c.pitchDeg = 0;
        c.distance = 10;
        QVector3D p = c.position();
        assert(near(p.x(), 0) && near(p.y(), 0) && near(p.z(), 10));
    }
    // yaw=90 -> +X axis.
    {
        Camera3D c;
        c.yawDeg = 90;
        c.pitchDeg = 0;
        c.distance = 10;
        QVector3D p = c.position();
        assert(near(p.x(), 10) && near(p.y(), 0) && near(p.z(), 0, 1e-3f));
    }
    // Pitch clamp: cannot flip over the poles.
    {
        Camera3D c;
        c.orbit(0, 10000);
        assert(c.pitchDeg <= 89.0f);
        c.orbit(0, -20000);
        assert(c.pitchDeg >= -89.0f);
    }
    // Grab-style orbit: drag right turns yaw left (scene follows cursor),
    // drag down raises the camera (top tips toward the viewer).
    {
        Camera3D c;
        const float y0 = c.yawDeg, p0 = c.pitchDeg;
        c.orbit(10, 0);
        assert(c.yawDeg < y0);
        c.orbit(0, 10);
        assert(c.pitchDeg > p0);
    }
    // Zoom clamp.
    {
        Camera3D c;
        c.zoomSteps(1000);
        assert(c.distance >= c.minDistance);
        c.zoomSteps(-1000);
        assert(c.distance <= c.maxDistance);
    }
    // View matrix looks at the target: target maps near screen center.
    {
        Camera3D c;
        c.distance = 18;
        QMatrix4x4 v = c.viewMatrix();
        QVector4D t = v * QVector4D(c.target, 1.0f);
        assert(near(t.x(), 0) && near(t.y(), 0) && t.z() < 0.0f);
    }
    // Layout: {2,8,1} with spacing 4 -> X slots -4,0,+4.
    {
        auto L = buildLayout3D({2, 8, 1}, 4.0f, 1.15f);
        assert(L.size() == 3);
        assert(L[0].positions.size() == 2);
        assert(L[1].positions.size() == 8);
        assert(L[2].positions.size() == 1);
        assert(near(L[0].positions[0].x(), -4.0f));
        assert(near(L[1].positions[0].x(), 0.0f));
        assert(near(L[2].positions[0].x(), 4.0f));
        // 8 neurons -> 3 cols x 3 rows grid.
        assert(L[1].cols == 3 && L[1].rows == 3);
        // Partial last row: mean within half a spacing of center.
        double my = 0, mz = 0;
        for (auto& p : L[1].positions) { my += p.y(); mz += p.z(); }
        assert(std::abs(my / 8.0) <= 0.575 && std::abs(mz / 8.0) <= 0.575);
        // Full grid centers exactly.
        {
            auto F = buildLayout3D({9}, 4.0f, 1.0f);
            double fy = 0, fz = 0;
            for (auto& p : F[0].positions) { fy += p.y(); fz += p.z(); }
            assert(near((float)(fy / 9.0), 0.0f) && near((float)(fz / 9.0), 0.0f));
        }
    }
    // 64 neurons -> 8x8 grid.
    {
        auto L = buildLayout3D({64}, 4.0f, 1.0f);
        assert(L[0].cols == 8 && L[0].rows == 8);
        assert(L[0].positions.size() == 64);
    }
    // Single-neuron layer sits at origin of its plane.
    {
        auto L = buildLayout3D({1}, 4.0f, 1.0f);
        assert(near(L[0].positions[0].x(), 0.0f) && near(L[0].positions[0].y(), 0.0f) &&
               near(L[0].positions[0].z(), 0.0f));
    }
    // Extent grows with depth and width.
    {
        float e1 = layoutExtent(buildLayout3D({2, 8, 1}));
        float e2 = layoutExtent(buildLayout3D({2, 64, 64, 1}));
        assert(e1 > 0 && e2 > e1);
    }
    // Camera lerp: endpoints exact, midpoint smooth, yaw takes short path.
    {
        Camera3D a, b;
        a.yawDeg = 170;
        b.yawDeg = -170; // 20 deg apart across the seam
        Camera3D m = lerpCam(a, b, 0.5f);
        assert(near(m.yawDeg, 180.0f) || near(m.yawDeg, -180.0f));
        Camera3D e0 = lerpCam(a, b, 0.0f), e1 = lerpCam(a, b, 1.0f);
        assert(near(e0.yawDeg, 170.0f) && near(e1.yawDeg, 190.0f)); // 190 == -170 mod 360
        assert(near(e0.distance, a.distance));
    }
    // Adaptive spacing: small nets airy, huge nets compact, never degenerate.
    {
        float small = adaptiveNeuronSpacing(8);
        float huge = adaptiveNeuronSpacing(784);
        assert(near(small, 1.15f) && huge < small && huge >= 0.45f);
        float ls = adaptiveLayerSpacing(small, 3);
        assert(ls >= 4.0f);
        float deep = adaptiveLayerSpacing(small, 10);
        assert(deep > ls);
    }
    // Parameter counter: 4->16->3 = 64+16 + 48+3 = 131.
    {
        assert(countParams({4, 16, 3}) == 131);
        assert(countParams({2, 8, 1}) == (2 * 8 + 8) + (8 * 1 + 1));
    }
    // Edge budget: small passes through, forced edges survive any cap,
    // overview keeps the strongest fraction when massively over budget.
    {
        std::vector<EdgeW> few = {{0, 0, 0, 0.5, false}, {0, 1, 0, 0.1, false}};
        assert(applyEdgeBudget(few, 20000).size() == 2);
        std::vector<EdgeW> many;
        for (int i = 0; i < 1000; ++i) many.push_back({0, i % 10, i / 10, 0.01 * (i % 7), false});
        many.push_back({0, 0, 0, 0.0001, true}); // forced weak edge
        auto kept = applyEdgeBudget(many, 100);
        assert(kept.size() <= 101); // 100 budget + forced kept
        bool hasForced = false;
        for (auto& e : kept)
            if (e.forced) hasForced = true;
        assert(hasForced);
        // Overview: 100k weak edges, cap 20k -> ~10k strongest kept.
        std::vector<EdgeW> flood;
        for (int i = 0; i < 100000; ++i) flood.push_back({0, 0, 0, 0.001 * (i % 1000), false});
        auto over = applyEdgeBudget(flood, 20000);
        assert(over.size() >= 9000 && over.size() <= 11000);
    }
    std::cout << "ALL VIZ3D-MATH CHECKS PASS\n";
    return 0;
}
