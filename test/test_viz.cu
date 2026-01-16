#include "visualization.cuh"
#include "loader.cuh"
#include <iostream>

int main() {
    Scene scene;
    
    scene.add_box(vec3(0, 0.4, 0.5), vec3(0.5, 0.5, 0.5), quat4(1, 0, 0, 0));
    scene.add_sphere(vec3(2, 0, 0.5), 0.5);
    scene.add_cylinder(vec3(-2, 0, 1), 0.3, 1.0, quat4(1, 0, 0, 0));
    
    quat4 rot = quat_from_axis_angle(vec3(0, 0, 1), M_PI / 4);
    scene.add_box(vec3(0, 2, 0.5), vec3(0.3, 0.3, 0.3), rot);

    std::cout << "Scene has " << scene.num_primitives() << " primitives" << std::endl;
    for (size_t i = 0; i < scene.primitives.size(); i++) {
        std::cout << "Primitive " << i << ": type = " << scene.primitives[i].type << std::endl;
    }
    
    std::vector<std::string> shader_files;
    shader_files.push_back("../visualization/shaders/vert.shader");
    shader_files.push_back("../visualization/shaders/frag.shader");

    Visualizer viz(1280, 720);
    if (!viz.init(shader_files, "SDF Scene Viewer")) {
        std::cerr << "Failed to initialize visualizer!" << std::endl;
        return 1;
    }
    
    viz.set_show_ground(true);
    viz.set_ground_height(0.0);
    viz.set_scene(&scene);
    
    // Run (blocking call)
    viz.run();
    
    return 0;
}