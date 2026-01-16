#ifndef CAMERA_H
#define CAMERA_H

#include "gl_include.cuh"
#include "math.cuh"

struct Camera {
    vec3 target;
    double distance;
    double yaw;
    double pitch;

    bool is_rotating;
    bool is_panning;
    double last_mouse_x;
    double last_mouse_y;

    Camera() :
        target(0, 0, 0), distance(5.0), yaw(45.0), pitch(30.0),
        is_rotating(false), is_panning(false),
        last_mouse_x(0), last_mouse_y(0)
    {}

    vec3 get_position() const;

    void get_view_matrix(float* mat) const;
    
    void on_mouse_button(int button, int action);
    void on_mouse_move(double xpos, double ypos);
    void on_scroll(double yoffset);
};

#endif