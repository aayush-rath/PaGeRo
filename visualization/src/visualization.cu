#include "visualization.cuh"
#include "math.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

char* readShaderFile(const char* filepath) {
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        std::cerr << "ERROR: Could not open shader file: " << filepath << std::endl;
        return nullptr;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Allocate buffer
    char* buffer = (char*)malloc(length + 1);
    if (!buffer) {
        fclose(file);
        return nullptr;
    }
    
    // Read file
    size_t read = fread(buffer, 1, length, file);
    buffer[read] = '\0';  // Null terminate
    
    fclose(file);
    
    printf("Loaded shader: %s (%ld bytes)\n", filepath, length);
    
    return buffer;
}

unsigned int compileShader(const char* source, GLenum type) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "[COMPILE ERROR]: " << infoLog << std::endl;
    }

    return shader;
}


Visualizer::Visualizer(int w, int h) :
    window(nullptr), width(w), height(h), scene(nullptr),
    show_grid(true), show_axes(true), show_wireframe(false),
    show_ground(true), shader_program(0),
    sphere_vao(0), sphere_vbo(0), sphere_ebo(0),
    box_vao(0), box_vbo(0), box_ebo(0),
    cylinder_vao(0), cylinder_vbo(0), cylinder_ebo(0),
    sphere_index_count(0), cylinder_index_count(0) 
{}

Visualizer::~Visualizer() {
    if (sphere_vao) glDeleteVertexArrays(1, &sphere_vao);
    if (sphere_vbo) glDeleteBuffers(1, &sphere_vbo);
    if (sphere_ebo) glDeleteBuffers(1, &sphere_ebo);

    if (box_vao) glDeleteVertexArrays(1, &box_vao);
    if (box_vbo) glDeleteBuffers(1, &box_vbo);
    if (box_ebo) glDeleteBuffers(1, &box_ebo);

    if (cylinder_vao) glDeleteVertexArrays(1, &cylinder_vao);
    if (cylinder_vbo) glDeleteBuffers(1, &cylinder_vbo);
    if (cylinder_ebo) glDeleteBuffers(1, &cylinder_ebo);

    if (shader_program) glDeleteProgram(shader_program);

    if (window) {
        glfwDestroyWindow(window);
    }
    glfwTerminate();
}

bool Visualizer::init(std::vector<std::string>& files, const char* title) {
    if (!glfwInit()) {
        std::cerr << "Couldn't initialize GLFW!\n";
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) {
        std::cerr << "Can't create a window!\n";
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);
    glfwSetWindowUserPointer(window, this);
    
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_button_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return false;
    }

    setup_opengl();
    setup_shaders(files[0], files[1]);

    create_sphere_mesh(10, 10);
    create_box_mesh();
    create_cylinder_mesh(10);

    std::cout << "Visualizer initialized successfully" << std::endl;
    std::cout << "\nControls:" << std::endl;
    std::cout << "  Left Mouse  : Rotate camera" << std::endl;
    std::cout << "  Right Mouse : Pan camera" << std::endl;
    std::cout << "  Scroll      : Zoom in/out" << std::endl;
    std::cout << "  G           : Toggle grid" << std::endl;
    std::cout << "  A           : Toggle axes" << std::endl;
    std::cout << "  W           : Toggle wireframe" << std::endl;
    std::cout << "  ESC         : Exit" << std::endl;
    
    return true;
}

void Visualizer::set_scene(Scene* s) {
    scene = s;
}

void Visualizer::setup_opengl() {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
}

void Visualizer::setup_shaders(std::string& vertex_shader_filename, std::string& fragment_shader_filename) {
    unsigned int vertex_shader = compileShader(readShaderFile(vertex_shader_filename.c_str()), GL_VERTEX_SHADER);
    unsigned int fragment_shader = compileShader(readShaderFile(fragment_shader_filename.c_str()), GL_FRAGMENT_SHADER);

    shader_program = glCreateProgram();
    glAttachShader(shader_program, vertex_shader);
    glAttachShader(shader_program, fragment_shader);
    glLinkProgram(shader_program);

    int success;
    glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shader_program, 512, nullptr, infoLog);
        std::cerr << "[LINK ERROR]: " << infoLog << std::endl;
    }

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
}

void Visualizer::create_sphere_mesh(int slices, int stacks) {
    std::vector<float> vertices;
    std::vector<unsigned int> indices;

    for (int i = 0; i <= stacks; i++) {
        float phi = M_PI * i / stacks;
        for (int j = 0; j <= slices; j++) {
            float theta = 2 * M_PI * j / slices;

            float x = sin(phi) * cos(theta);
            float y = sin(phi) * sin(theta);
            float z = cos(phi);

            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);

            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        }
    }

    for (int i = 0; i < stacks; i++) {
        for (int j = 0; j < slices; j++) {
            int first = i * (slices + 1) + j;
            int second = first + slices + 1;

            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);

            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
        }
    }

    sphere_index_count = indices.size();

    glGenVertexArrays(1, &sphere_vao);
    glGenBuffers(1, &sphere_vbo);
    glGenBuffers(1, &sphere_ebo);

    glBindVertexArray(sphere_vao);
    
    glBindBuffer(GL_ARRAY_BUFFER, sphere_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices.size(), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphere_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices.size(), indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void Visualizer::create_box_mesh() {
    // Cube vertices with normals
    float vertices[] = {
        // Positions          // Normals
        // Front face (Z+)
        -1, -1,  1,   0,  0,  1,
         1, -1,  1,   0,  0,  1,
         1,  1,  1,   0,  0,  1,
        -1,  1,  1,   0,  0,  1,
        // Back face (Z-)
        -1, -1, -1,   0,  0, -1,
         1, -1, -1,   0,  0, -1,
         1,  1, -1,   0,  0, -1,
        -1,  1, -1,   0,  0, -1,
        // Left face (X-)
        -1, -1, -1,  -1,  0,  0,
        -1, -1,  1,  -1,  0,  0,
        -1,  1,  1,  -1,  0,  0,
        -1,  1, -1,  -1,  0,  0,
        // Right face (X+)
         1, -1, -1,   1,  0,  0,
         1, -1,  1,   1,  0,  0,
         1,  1,  1,   1,  0,  0,
         1,  1, -1,   1,  0,  0,
        // Top face (Z+)
        -1,  1, -1,   0,  1,  0,
         1,  1, -1,   0,  1,  0,
         1,  1,  1,   0,  1,  0,
        -1,  1,  1,   0,  1,  0,
        // Bottom face (Y-)
        -1, -1, -1,   0, -1,  0,
         1, -1, -1,   0, -1,  0,
         1, -1,  1,   0, -1,  0,
        -1, -1,  1,   0, -1,  0
    };
    
    unsigned int indices[] = {
        0, 1, 2, 2, 3, 0,       // Front
        4, 5, 6, 6, 7, 4,       // Back
        8, 9, 10, 10, 11, 8,    // Left
        12, 13, 14, 14, 15, 12, // Right
        16, 17, 18, 18, 19, 16, // Top
        20, 21, 22, 22, 23, 20  // Bottom
    };
    
    glGenVertexArrays(1, &box_vao);
    glGenBuffers(1, &box_vbo);
    glGenBuffers(1, &box_ebo);
    
    glBindVertexArray(box_vao);
    
    glBindBuffer(GL_ARRAY_BUFFER, box_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, box_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), 
                         (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
}

void Visualizer::create_cylinder_mesh(int slices) {
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    
    for (int i = 0; i <= slices; ++i) {
        float theta = 2 * M_PI * i / slices;
        float x = cos(theta);
        float y = sin(theta);
        
        vertices.push_back(x);
        vertices.push_back(y);
        vertices.push_back(1.0f);
        vertices.push_back(x);
        vertices.push_back(y);
        vertices.push_back(0);
        
        vertices.push_back(x);
        vertices.push_back(y);
        vertices.push_back(-1.0f);
        vertices.push_back(x);
        vertices.push_back(y);
        vertices.push_back(0);
    }

    for (int i = 0; i < slices; ++i) {
        int top1 = i * 2;
        int bottom1 = i * 2 + 1;
        int top2 = (i + 1) * 2;
        int bottom2 = (i + 1) * 2 + 1;
        
        indices.push_back(top1);
        indices.push_back(bottom1);
        indices.push_back(top2);
        
        indices.push_back(bottom1);
        indices.push_back(bottom2);
        indices.push_back(top2);
    }
    
    int top_center_idx = vertices.size() / 6;
    vertices.push_back(0); vertices.push_back(0); vertices.push_back(1);
    vertices.push_back(0); vertices.push_back(0); vertices.push_back(1);
    
    int bottom_center_idx = vertices.size() / 6;
    vertices.push_back(0); vertices.push_back(0); vertices.push_back(-1);
    vertices.push_back(0); vertices.push_back(0); vertices.push_back(-1);
    
    for (int i = 0; i < slices; ++i) {
        indices.push_back(top_center_idx);
        indices.push_back(i * 2);
        indices.push_back((i + 1) * 2);
    }
    
    for (int i = 0; i < slices; ++i) {
        indices.push_back(bottom_center_idx);
        indices.push_back((i + 1) * 2 + 1);
        indices.push_back(i * 2 + 1);
    }
    
    cylinder_index_count = indices.size();
    
    glGenVertexArrays(1, &cylinder_vao);
    glGenBuffers(1, &cylinder_vbo);
    glGenBuffers(1, &cylinder_ebo);
    
    glBindVertexArray(cylinder_vao);
    
    glBindBuffer(GL_ARRAY_BUFFER, cylinder_vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float),
                 vertices.data(), GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cylinder_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
                 indices.data(), GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                         (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
}

void Visualizer::render_grid() {
    if (!show_grid) return;

    float grid_size = 20.0f;
    int divisions =  20;
    float step = grid_size / divisions;

    std::vector<float> vertices;

    for (int i = 0; i <= divisions; ++i) {
        float pos = -grid_size/2 + i * step;
        
        // Line along X
        vertices.push_back(-grid_size/2); vertices.push_back(pos); vertices.push_back(0);
        vertices.push_back(0); vertices.push_back(0); vertices.push_back(1);
        vertices.push_back(grid_size/2);  vertices.push_back(pos); vertices.push_back(0);
        vertices.push_back(0); vertices.push_back(0); vertices.push_back(1);
        
        // Line along Y
        vertices.push_back(pos); vertices.push_back(-grid_size/2); vertices.push_back(0);
        vertices.push_back(0); vertices.push_back(0); vertices.push_back(1);
        vertices.push_back(pos); vertices.push_back(grid_size/2);  vertices.push_back(0);
        vertices.push_back(0); vertices.push_back(0); vertices.push_back(1);
    }

    unsigned int grid_vao, grid_vbo;
    glGenVertexArrays(1, &grid_vao);
    glGenBuffers(1, &grid_vbo);
    
    glBindVertexArray(grid_vao);
    glBindBuffer(GL_ARRAY_BUFFER, grid_vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float),
                 vertices.data(), GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                         (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    // Identity model matrix
    float model[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, model);
    glUniform3f(glGetUniformLocation(shader_program, "objectColor"), 0.3f, 0.3f, 0.3f);
    
    glDrawArrays(GL_LINES, 0, vertices.size() / 6);
    
    glDeleteBuffers(1, &grid_vbo);
    glDeleteVertexArrays(1, &grid_vao);
}

void Visualizer::render_axes() {
    if (!show_axes) return;
    
    float vertices[] = {
        // X axis (red)
        0, 0, 0,  1, 0, 0,
        1, 0, 0,  1, 0, 0,
        // Y axis (green)
        0, 0, 0,  0, 1, 0,
        0, 1, 0,  0, 1, 0,
        // Z axis (blue)
        0, 0, 0,  0, 0, 1,
        0, 0, 1,  0, 0, 1
    };
    
    unsigned int axes_vao, axes_vbo;
    glGenVertexArrays(1, &axes_vao);
    glGenBuffers(1, &axes_vbo);
    
    glBindVertexArray(axes_vao);
    glBindBuffer(GL_ARRAY_BUFFER, axes_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                         (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    float model[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, model);
    
    glUniform3f(glGetUniformLocation(shader_program, "objectColor"), 1, 0, 0);
    glDrawArrays(GL_LINES, 0, 2);

    glUniform3f(glGetUniformLocation(shader_program, "objectColor"), 0, 1, 0);
    glDrawArrays(GL_LINES, 2, 2);

    glUniform3f(glGetUniformLocation(shader_program, "objectColor"), 0, 0, 1);
    glDrawArrays(GL_LINES, 4, 2);
    
    glDeleteBuffers(1, &axes_vbo);
    glDeleteVertexArrays(1, &axes_vao);
}

void Visualizer::render_ground() {
    if (!show_ground) return;
    
    // Draw a large grid plane
    glUseProgram(shader_program);
    
    // Set ground color (gray)
    glUniform3f(glGetUniformLocation(shader_program, "objectColor"), 
                0.3f, 0.3f, 0.3f);
    
    // Identity model matrix (no transform)
    float model[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, ground_height, 1
    };
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 
                       1, GL_FALSE, model);
    
    float grid_size = 50.0f;
    int divisions = 50;
    
    std::vector<float> vertices;
    
    // Generate grid lines
    for (int i = 0; i <= divisions; i++) {
        float t = -grid_size/2 + (i * grid_size / divisions);
        
        // Line along X
        vertices.push_back(-grid_size/2); vertices.push_back(t); vertices.push_back(0);
        vertices.push_back(0); vertices.push_back(0); vertices.push_back(1); // Normal
        vertices.push_back(grid_size/2);  vertices.push_back(t); vertices.push_back(0);
        vertices.push_back(0); vertices.push_back(0); vertices.push_back(1);
        
        // Line along Y
        vertices.push_back(t); vertices.push_back(-grid_size/2); vertices.push_back(0);
        vertices.push_back(0); vertices.push_back(0); vertices.push_back(1);
        vertices.push_back(t); vertices.push_back(grid_size/2);  vertices.push_back(0);
        vertices.push_back(0); vertices.push_back(0); vertices.push_back(1);
    }
    
    // Create VAO/VBO for ground
    unsigned int ground_vao, ground_vbo;
    glGenVertexArrays(1, &ground_vao);
    glGenBuffers(1, &ground_vbo);
    
    glBindVertexArray(ground_vao);
    glBindBuffer(GL_ARRAY_BUFFER, ground_vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), 
                 vertices.data(), GL_STATIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), 
                         (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    // Draw lines
    glDrawArrays(GL_LINES, 0, vertices.size() / 6);
    
    glDeleteBuffers(1, &ground_vbo);
    glDeleteVertexArrays(1, &ground_vao);
}

void Visualizer::draw_sphere(const Sphere& sphere, const vec3& color) {
    float model[16];
    quat_to_mat(quat4(1, 0, 0, 0), model);

    for (int i = 0; i < 3; i++) {
        model[i * 4 + 0] *= sphere.radius;
        model[i * 4 + 1] *= sphere.radius;
        model[i * 4 + 2] *= sphere.radius;
    }

    model[12] = sphere.center.x();
    model[13] = sphere.center.y();
    model[14] = sphere.center.z();

    glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, model);
    glUniform3f(glGetUniformLocation(shader_program, "objectColor"), color.x(), color.y(), color.z());
    glBindVertexArray(sphere_vao);
    glDrawElements(GL_TRIANGLES, sphere_index_count, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void Visualizer::draw_box(const Box& box, const vec3& color) {
    float model[16];
    quat_to_mat(box.orientation, model);

    for (int i = 0; i < 3; i++) {
        model[i * 4 + 0] *= box.size.x() / 2.0;
        model[i * 4 + 1] *= box.size.y() / 2.0;
        model[i * 4 + 2] *= box.size.z() / 2.0;
    }

    model[12] = box.center.x();
    model[13] = box.center.y();
    model[14] = box.center.z();

    glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, model);
    glUniform3f(glGetUniformLocation(shader_program, "objectColor"), color.x(), color.y(), color.z());
    glBindVertexArray(box_vao);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void Visualizer::draw_cylinder(const Cylinder& cylinder, const vec3& color) {
    float model[16];
    quat_to_mat(cylinder.orientation, model);

    model[0] *= cylinder.radius;
    model[1] *= cylinder.radius;
    model[2] *= cylinder.radius;

    model[4] *= cylinder.radius;
    model[5] *= cylinder.radius;
    model[6] *= cylinder.radius;

    model[8] *= cylinder.height / 2;
    model[9] *= cylinder.height / 2;
    model[10] *= cylinder.height / 2;

    model[12] = cylinder.center.x();
    model[13] = cylinder.center.y();
    model[14] = cylinder.center.z();

    glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, model);
    glUniform3f(glGetUniformLocation(shader_program, "objectColor"), color.x(), color.y(), color.z());
    glBindVertexArray(cylinder_vao);
    glDrawElements(GL_TRIANGLES, cylinder_index_count, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void Visualizer::render() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(shader_program);

    float view[16], projection[16];
    camera.get_view_matrix(view);

    float aspect = (float)width / (float)height;
    float fov = 45.0f * M_PI / 180.0f;
    float near_plane = 0.1f, far_plane = 100.0f;
    float f = 1.0f / tan(fov / 2.0f);

    projection[0] = f/aspect; projection[4] = 0; projection[8]  = 0;                              projection[12] = 0;
    projection[1] = 0;        projection[5] = f; projection[9]  = 0;                              projection[13] = 0;
    projection[2] = 0;        projection[6] = 0; projection[10] = -(far_plane+near_plane)/(far_plane-near_plane); projection[14] = -(2*far_plane*near_plane)/(far_plane-near_plane);
    projection[3] = 0;        projection[7] = 0; projection[11] = -1;                             projection[15] = 0;

    glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, view);
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, projection);

    vec3 cam_pos = camera.get_position();
    glUniform3f(glGetUniformLocation(shader_program, "lightPos"), 10, 10, 10);
    glUniform3f(glGetUniformLocation(shader_program, "viewPos"), 
                cam_pos.x(), cam_pos.y(), cam_pos.z());

    render_ground();
    render_grid();
    render_axes();
    render_robot();

    if (scene) {
        if (show_wireframe) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        }

        for (const auto& prim : scene->primitives) {
            vec3 color(0.6, 0.6, 0.8);
            
            switch (prim.type) {
                case PRIM_SPHERE:
                    draw_sphere(prim.data.sphere, color);
                    break;
                case PRIM_BOX:
                    draw_box(prim.data.box, color);
                    break;
                case PRIM_CYLINDER:
                    draw_cylinder(prim.data.cylinder, color);
                    break;
            }
        }

        if (show_wireframe) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }
    }
}

void Visualizer::run() {
    while(!glfwWindowShouldClose(window)) {
        render();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

void Visualizer::set_robot(Kinematics* kinematics) {
    robot_kinematics = kinematics;
    vec3 color(1.0, 0.8, 0.8);
    num_dof = kinematics->get_robot().num_dof();
    active_joint = 0;

    show_robot = true;
    update_robot();
}

void Visualizer::update_robot() {
    if (robot_kinematics) {
        robot_primitives = robot_kinematics->get_transformed_primitives();
    }
}

void Visualizer::clear_robot() {
    robot_kinematics = nullptr;
    robot_primitives.clear();
    show_robot = false;
}

void Visualizer::render_robot() {
    std::vector<vec3> link_colors = {
        vec3(0.8, 0.3, 0.3),
        vec3(0.3, 0.8, 0.3),
        vec3(0.3, 0.3, 0.8),
        vec3(0.8, 0.8, 0.3),
        vec3(0.8, 0.3, 0.8),
        vec3(0.3, 0.8, 0.8),
    };
    if (show_wireframe) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }
    for (int i = 0; i < robot_primitives.size(); i++) {
        draw_robot_primitive(robot_primitives[i], link_colors[i]);
    }
}

void Visualizer::draw_robot_primitive(const Primitive& primitive, const vec3& color) {
    switch (primitive.type) {
    case PRIM_SPHERE:
        draw_sphere(primitive.data.sphere, color);
        break;

    case PRIM_BOX:
        draw_box(primitive.data.box, color);
        break;   

    case PRIM_CYLINDER:
        draw_cylinder(primitive.data.cylinder, color);
        break;
    }
}

void Visualizer::set_active_joint(int joint_idx) {
    active_joint = joint_idx;
}

int Visualizer::get_active_joint() const {
    return active_joint;
}

int Visualizer::get_num_dof() const {
    return num_dof;
}

void Visualizer::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    Visualizer *viz = static_cast<Visualizer*>(glfwGetWindowUserPointer(window));
    viz->camera.on_mouse_button(button, action);
}

void Visualizer::cursor_button_callback(GLFWwindow* window, double xpos, double ypos) {
    Visualizer *viz = static_cast<Visualizer*>(glfwGetWindowUserPointer(window));
    viz->camera.on_mouse_move(xpos, ypos);
}

void Visualizer::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    Visualizer *viz = static_cast<Visualizer*>(glfwGetWindowUserPointer(window));
    viz->camera.on_scroll(yoffset);
}

void Visualizer::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    Visualizer *viz = static_cast<Visualizer*>(glfwGetWindowUserPointer(window));

    switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, true);
            break;
        case GLFW_KEY_G:
            if (action == GLFW_PRESS) {viz->show_grid = !viz->show_grid;
            std::cout << "Grid: " << (viz->show_grid ? "ON" : "OFF") << std::endl;}
            break;
        case GLFW_KEY_A:
            if (action == GLFW_PRESS) {viz->show_axes = !viz->show_axes;
            std::cout << "Axes: " << (viz->show_axes ? "ON" : "OFF") << std::endl;}
            break;
        case GLFW_KEY_W:
            if (action == GLFW_PRESS) {viz->show_wireframe = !viz->show_wireframe;
            std::cout << "Wireframe: " << (viz->show_wireframe ? "ON" : "OFF") << std::endl;}
            break;
        case GLFW_KEY_LEFT_BRACKET:
            if (action == GLFW_PRESS) {if (viz->get_active_joint() != 0) viz->set_active_joint(viz->get_active_joint() - 1);
            std::cout << "Current Active Joint: " << viz->get_active_joint() << std::endl;}
            break;
        case GLFW_KEY_RIGHT_BRACKET:
            if (action == GLFW_PRESS) {if (viz->get_active_joint() != viz->get_num_dof() - 1) viz->set_active_joint(viz->get_active_joint() + 1);
            std::cout << "Current Active Joint: " << viz->get_active_joint() << std::endl;}
            break;
        case GLFW_KEY_MINUS:
            viz->robot_kinematics->set_joint_position(viz->get_active_joint(), viz->robot_kinematics->get_joint_positions()[viz->get_active_joint()] - 0.05);
            viz->robot_kinematics->update_forward_kinematics();
            viz->update_robot();
            break;
        case GLFW_KEY_EQUAL:
            viz->robot_kinematics->set_joint_position(viz->get_active_joint(), viz->robot_kinematics->get_joint_positions()[viz->get_active_joint()] + 0.05);
            viz->robot_kinematics->update_forward_kinematics();
            viz->update_robot();
            break;
    }
}

void Visualizer::framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    Visualizer* viz = static_cast<Visualizer*>(glfwGetWindowUserPointer(window));
    viz->width = width;
    viz->height = height;
}