#include <iostream>
#include <vector>
#include <cmath>
#include "geometry.cuh"
#include "loader.cuh"

int main() {
    Scene scene;
    
    scene.add_box(vec3(0, 0.4, 0.5), vec3(0.5, 0.5, 0.5), quat4(1, 0, 0, 0));
    scene.add_sphere(vec3(2, 0, 0.5), 0.5);
    scene.add_cylinder(vec3(-2, 0, 1), 0.3, 1.0, quat4(1, 0, 0, 0));
    
    quat4 rot = quat_from_axis_angle(vec3(0, 0, 1), M_PI / 4);
    scene.add_box(vec3(0, 2, 0.5), vec3(0.3, 0.3, 0.3), rot);

    // Test points for SDF evaluation
    std::vector<vec3> test_points = {
        vec3(0, 0.4, 0.5),    // inside first box
        vec3(2, 0, 0.5),      // center of sphere
        vec3(-2, 0, 1),       // center of cylinder
        vec3(0, 2, 0.5),      // center of rotated box
        vec3(1, 1, 1),        // outside all
        vec3(0, 0, 0)         // near origin
    };

    std::cout << "Using point: " << test_points[0].x() << " " << test_points[0].y() << " " << test_points[0].z() << std::endl; 

    for (int i = 0; i < scene.primitives.size(); i++) {
        double distance = scene.primitives[i].sdf(test_points[0]);
        std::cout << "Primitive " << i << ": distance = " << distance << std::endl;
    }

    return 0;
}