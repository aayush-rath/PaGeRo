#include "visualization.cuh"


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <urdf_file> <scene_file>" << std::endl;
        return -1;
    }

    Robot robot = load_urdf(argv[1]);
    Kinematics kinematics(robot);

    std::cout << "Joint type: " << (robot.joints[0].type == PRISMATIC ? "Prismatic" : "Revolute") << std::endl;

    Visualizer visualizer(1280, 720);
    std::vector<std::string> shader_files = {"../visualization/shaders/vert.shader", "../visualization/shaders/frag.shader"};
    visualizer.init(shader_files);

    Scene scene = load_scene_json(argv[2]);

    visualizer.set_scene(&scene);
    visualizer.set_robot(&kinematics);

    visualizer.set_show_ground(true);
    visualizer.set_ground_height(0.0);
    visualizer.set_scene(&scene);
    
    visualizer.run();

    return 0;
}