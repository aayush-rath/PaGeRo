#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include "gl_include.cuh"
#include "geometry.cuh"
#include "loader.cuh"
#include "camera.cuh"

class Visualizer {
public:
    Visualizer(int width = 1280, int height = 720);
    ~Visualizer();

    bool init(std::vector<std::string>& files, const char* title = "SDF Visualizer");

    void set_scene(Scene* scene);
    void run();

    void set_show_grid(bool show) {show_grid = show; }
    void set_show_axes(bool show) {show_axes = show; }
    void set_show_wireframe(bool show) {show_axes = show; }
    void set_show_ground(bool show) {show_ground = show; }
    void set_ground_height(double height) {ground_height = height; }

private:
    GLFWwindow* window;
    int width, height;

    Scene* scene;

    Camera camera;

    bool show_grid;
    bool show_axes;
    bool show_wireframe;
    bool show_ground;
    double ground_height;

    unsigned int shader_program;
    unsigned int sphere_vao, sphere_vbo, sphere_ebo;
    unsigned int cylinder_vao, cylinder_vbo, cylinder_ebo;
    unsigned int box_vao, box_vbo, box_ebo;

    int sphere_index_count;
    int cylinder_index_count;

    void setup_opengl();
    void setup_shaders(std::string& vertex_shader_filename, std::string& fragment_shader_filename);
    void create_sphere_mesh(int slices, int stacks);
    void create_box_mesh();
    void create_cylinder_mesh(int slices);

    void render();
    void render_grid();
    void render_axes();
    void render_ground();
    void draw_box(const Box& box, const vec3& color);
    void draw_sphere(const Sphere& sphere, const vec3& color);
    void draw_cylinder(const Cylinder& cylinder, const vec3& color);

    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void cursor_button_callback(GLFWwindow* window, double xpos, double ypos);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
};

#endif