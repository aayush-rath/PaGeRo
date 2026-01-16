#pragma once
#include "math.cuh"
#include <algorithm>

__host__ __device__
inline double clamp(double x, double a, double b) {
    return max(a, min(b, x));
}

__host__ __device__
inline double length(const vec3& v) {
    return sqrt(dot(v, v));
}


struct Plane {
    vec3 point;
    vec3 normal;

    __host__ __device__
    double sdf(const vec3& p) const {
        return dot(p - point, normal);
    }
};

struct Box{
    vec3 center;
    quat4 orientation;
    vec3 size;

    __host__ __device__
    double sdf(const vec3& p) const {
        vec3 p_local = rotate(p - center, orientation.inverse());

        double dx = abs(p_local.x()) - size.x() / 2.0;
        double dy = abs(p_local.y()) - size.y() / 2.0; 
        double dz = abs(p_local.z()) - size.z() / 2.0;

        double max_coord = max(dx, max(dy, dz));

        double dist = sqrt(max(dx, 0.0) * max(dx, 0.0) + max(dy, 0.0) * max(dy, 0.0) + max(dz, 0.0) * max(dz, 0.0));
        return dist + min(max_coord, 0.0);
    }
};

struct Sphere {
    vec3 center;
    double radius;

    __host__  __device__
    double sdf(const vec3& p) const {
        return norm(p, center) - radius;
    }
};

struct Cylinder {
    vec3 center;
    double radius;
    double height;
    quat4 orientation;

    __host__ __device__
    double sdf(const vec3& p) const {
        vec3 p_local = rotate(p - center, orientation.inverse());
        
        double r = sqrt(p_local.x() * p_local.x() + p_local.y() * p_local.y());

        double dx = r - radius;
        double dy = abs(p_local.z()) - height / 2.0;

        double outside = sqrt(max(dx, 0.0) * max(dx, 0.0) + 
                             max(dy, 0.0) * max(dy, 0.0));

        double inside = min(max(dx, dy), 0.0);
        return outside + inside;
    }
};

enum PrimitiveType{
    PRIM_SPHERE = 0,
    PRIM_BOX = 1,
    PRIM_CYLINDER = 2
};

union PrimitiveData {
    Sphere sphere;
    Box box;
    Cylinder cylinder;
};

struct Primitive {
    PrimitiveType type;
    PrimitiveData data;

    __host__ __device__
    double sdf(const vec3& p) const {
        switch (type) {
            case PRIM_SPHERE: return data.sphere.sdf(p);
            case PRIM_BOX: return data.box.sdf(p);
            case PRIM_CYLINDER: return data.cylinder.sdf(p);
        }
    }
};

__host__ __device__
double distance(const Sphere& a, const Sphere& b) {
    double d = length(b.center - a.center);
    return d - (a.radius + b.radius);
}

__host__ __device__
double distance(const Cylinder& a, const Cylinder& b) {
    vec3 za = rotate(vec3(0,0,1), a.orientation);
    vec3 zb = rotate(vec3(0,0,1), b.orientation);

    vec3 pa = a.center;
    vec3 pb = b.center;

    vec3 d = pb - pa;

    double da = dot(d, za);
    double db = dot(d, zb);

    double ha = a.height * 0.5;
    double hb = b.height * 0.5;

    da = clamp(da, -ha, ha);
    db = clamp(db, -hb, hb);

    vec3 ca = pa + za * da;
    vec3 cb = pb + zb * db;

    double centerDist = length(ca - cb);
    return centerDist - (a.radius + b.radius);
}

__host__ __device__
double distance(const Box& a, const Box& b) {
    vec3 axes[3] = {
        rotate(vec3(1,0,0), a.orientation),
        rotate(vec3(0,1,0), a.orientation),
        rotate(vec3(0,0,1), a.orientation)
    };

    double minDist = 1e30;

    for (int i = 0; i < 3; ++i) {
        vec3 p = a.center + axes[i] * (a.size[i] * 0.5);
        minDist = min(minDist, b.sdf(p));
        p = a.center - axes[i] * (a.size[i] * 0.5);
        minDist = min(minDist, b.sdf(p));
    }

    return minDist;
}


__host__ __device__
double distance(const Sphere& s, const Box& b) {
    vec3 p = rotate(s.center - b.center, b.orientation.inverse());
    vec3 h = b.size * 0.5;

    vec3 q(
        clamp(p.x(), -h.x(), h.x()),
        clamp(p.y(), -h.y(), h.y()),
        clamp(p.z(), -h.z(), h.z())
    );

    double d = length(p - q);
    return d - s.radius;
}

__host__ __device__
double distance(const Sphere& s, const Cylinder& c) {
    vec3 p = rotate(s.center - c.center, c.orientation.inverse());

    double r_xy = sqrt(p.x()*p.x() + p.y()*p.y());
    double dz   = abs(p.z()) - c.height * 0.5;

    double dx = r_xy - c.radius;

    double outside = sqrt(max(dx, 0.0)*max(dx, 0.0)
                        + max(dz, 0.0)*max(dz, 0.0));

    double inside = min(max(dx, dz), 0.0);

    return outside + inside - s.radius;
}

__host__ __device__
double distance(const Box& b, const Cylinder& c) {
    double minDist = 1e30;

    // Box vertices (8)
    for (int sx = -1; sx <= 1; sx += 2)
    for (int sy = -1; sy <= 1; sy += 2)
    for (int sz = -1; sz <= 1; sz += 2) {
        vec3 v = vec3(sx, sy, sz) * (b.size * 0.5);
        vec3 p = b.center + rotate(v, b.orientation);
        minDist = min(minDist, c.sdf(p));
    }

    // Cylinder axis endpoints
    vec3 z = rotate(vec3(0,0,1), c.orientation);
    minDist = min(minDist, b.sdf(c.center + z * (c.height * 0.5)));
    minDist = min(minDist, b.sdf(c.center - z * (c.height * 0.5)));

    return minDist;
}
