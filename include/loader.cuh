#pragma once
#include "geometry.cuh"
#include "robot.cuh"
#include "kinematics.cuh"
#include <vector>

struct Scene {
    std::vector<Primitive> primitives;
    
    Primitive* send_primitives_to_device() const;
    int num_primitives() const { return primitives.size(); }

    void add_sphere(vec3 center, double radius, vec3 color);
    void add_box(vec3 center, vec3 size, quat4 orientation, vec3 color);
    void add_cylinder(vec3 center, double radius, double half_height, quat4 orientation, vec3 color);
};


Scene load_scene_json(const char* filename);
Robot load_urdf(const char* filename);


