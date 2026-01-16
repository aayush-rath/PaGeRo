#include "camera.cuh"
#include "math.cuh"

vec3 Camera::get_position() const {
    double yaw_rad = yaw * M_PI / 180.0;
    double pitch_rad = pitch * M_PI / 180.0;
    
    double x = distance * cos(pitch_rad) * cos(yaw_rad);
    double y = distance * cos(pitch_rad) * sin(yaw_rad);
    double z = distance * sin(pitch_rad);

    return vec3(target.x() + x, target.y() + y, target.z() + z);
}

void Camera::get_view_matrix(float* mat) const {
    vec3 pos = get_position();
    vec3 forward = target - pos;

    forward = forward / forward.norm();

    vec3 world_up(0, 0, 1);

    vec3 right = cross(forward, world_up);
    right = right / right.norm();

    vec3 up = cross(right, forward);
    mat[0] = right.x();   mat[4] = right.y();   mat[8]  = right.z();   mat[12] = -dot(right, pos);
    mat[1] = up.x();      mat[5] = up.y();      mat[9]  = up.z();      mat[13] = -dot(up, pos);
    mat[2] = -forward.x(); mat[6] = -forward.y(); mat[10] = -forward.z(); mat[14] = dot(forward, pos);
    mat[3] = 0;           mat[7] = 0;           mat[11] = 0;           mat[15] = 1;
}

void Camera::on_mouse_button(int button, int action) {
    if (button == 0) {
        is_rotating = (action == 1);
    } else if (button == 1) {
        is_panning = (action == 1);
    }
}

void Camera::on_mouse_move(double xpos, double ypos) {
    double dx = xpos - last_mouse_x;
    double dy = ypos - last_mouse_y;

    if (is_rotating) {

        yaw += dx * 0.5;
        pitch -= std::max(-89.0, std::min(89.0, dy * 0.5));

    } else if (is_panning) {
        double speed = distance * 0.001;

        double yaw_rad = yaw * M_PI / 180.0;
        vec3 right(cos(yaw_rad), sin(yaw_rad), 0);
        vec3 up(-sin(yaw_rad), cos(yaw_rad), 0);

        target = target - right * (dy * speed);
        target = target - up * (dx * speed);
    }

    last_mouse_x = xpos;
    last_mouse_y = ypos;
}

void Camera::on_scroll(double yoffset) {
    distance -= yoffset * 0.5;
    distance = std::max(0.5, std::min(50.0, distance));
}