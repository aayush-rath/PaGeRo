#include "visualization.cuh"


int main() {
    Robot robot = load_urdf("../robots/3DOFRoboticArm.urdf");
    Kinematics kinematics(robot);

    std::cout << "Joint type: " << (robot.joints[0].type == PRISMATIC ? "Prismatic" : "Revolute") << std::endl;

    Visualizer visualizer(1280, 720);
    std::vector<std::string> shader_files = {"../visualization/shaders/vert.shader", "../visualization/shaders/frag.shader"};
    visualizer.init(shader_files);

    Scene scene;
    scene.add_sphere(vec3(2.0, 0.0, 0.5), 0.3);
    scene.add_box(vec3(-1.5, 0.0, 0.5), vec3(0.4, 0.4, 0.4), quat4(1, 0, 0, 0));
    scene.add_cylinder(vec3(0.0, 2.0, 0.5), 0.2, 0.8, quat4(1, 0, 0, 0));

    visualizer.set_scene(&scene);
    visualizer.set_robot(&kinematics);

    visualizer.set_show_ground(true);
    visualizer.set_ground_height(0.0);
    visualizer.set_scene(&scene);
    
    visualizer.run();

    return 0;
}