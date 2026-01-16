cmake_minimum_required(VERSION 3.16)
project(geom LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80)

# ============================================================================
# Include Directories
# ============================================================================

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/visualization/include)

# ============================================================================
# Find OpenGL, GLEW, GLFW (for visualization)
# ============================================================================

find_package(OpenGL QUIET)
find_package(GLEW QUIET)
find_package(glfw3 QUIET)

set(BUILD_VISUALIZER OFF)
if(OpenGL_FOUND AND GLEW_FOUND AND glfw3_FOUND)
    set(BUILD_VISUALIZER ON)
    message(STATUS "OpenGL, GLEW, and GLFW3 found - visualizer will be built")
else()
    message(STATUS "OpenGL, GLEW, or GLFW3 not found - visualizer will NOT be built")
    message(STATUS "To build visualizer, install dependencies:")
    message(STATUS "  sudo apt-get install libglfw3-dev libglew-dev libgl1-mesa-dev")
endif()

find_package(tinyxml2 REQUIRED)
find_package(nlohmann_json REQUIRED)

# ============================================================================
# Core Library (loader.cu + geometry)
# ============================================================================

add_library(geom_core STATIC
    ${CMAKE_SOURCE_DIR}/src/loader.cu
)

target_include_directories(geom_core PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
    ${TinyXML2_INCLUDE_DIRS}
    ${nlohmann_json_INCLUDE_DIRS}
)
target_link_libraries(geom_core
    PUBLIC
        tinyxml2::tinyxml2
        nlohmann_json::nlohmann_json
)
set_target_properties(geom_core PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# SDF Test Executable
# ============================================================================

add_executable(test_sdf
    ${CMAKE_SOURCE_DIR}/test/test_sdf.cu
)

target_include_directories(test_sdf PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_sdf PRIVATE geom_core)
set_target_properties(test_sdf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ============================================================================
# Loading Test Executable
# ============================================================================

add_executable(test_loading
    ${CMAKE_SOURCE_DIR}/test/test_loading.cu
)

target_include_directories(test_loading PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_loading PRIVATE geom_core)
set_target_properties(test_loading PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ============================================================================
# Visualization Library (Only if dependencies found)
# ============================================================================

if(BUILD_VISUALIZER)
    # Visualization sources
    set(VIZ_SOURCES
        ${CMAKE_SOURCE_DIR}/visualization/src/visualization.cu
        ${CMAKE_SOURCE_DIR}/visualization/src/camera.cu
    )
    
    # Create visualization library
    add_library(viz_lib STATIC ${VIZ_SOURCES})
    
    target_include_directories(viz_lib 
        PUBLIC 
            ${CMAKE_SOURCE_DIR}/visualization/include
            ${CMAKE_SOURCE_DIR}/include
            ${TinyXML2_INCLUDE_DIRS}
            ${nlohmann_json_INCLUDE_DIRS}
    )
    
    target_link_libraries(viz_lib
        PUBLIC
            OpenGL::GL
            GLEW::GLEW
            glfw
            tinyxml2::tinyxml2
            nlohmann_json::nlohmann_json
    )
    
    set_target_properties(viz_lib PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    
    # ========================================================================
    # Visualization Test Executable
    # ========================================================================
    
    add_executable(test_viz
        ${CMAKE_SOURCE_DIR}/test/test_viz.cu
    )
    
    target_include_directories(test_viz 
        PRIVATE 
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/visualization/include
            ${TinyXML2_INCLUDE_DIRS}
            ${nlohmann_json_INCLUDE_DIRS}
    )
    
    target_link_libraries(test_viz
        PRIVATE
            geom_core      # ADD THIS - links loader.cu
            viz_lib
            OpenGL::GL
            GLEW::GLEW
            glfw
            tinyxml2::tinyxml2
            nlohmann_json::nlohmann_json
    )
    
    set_target_properties(test_viz PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    
    # Copy shaders to build directory
    file(COPY ${CMAKE_SOURCE_DIR}/visualization/shaders
         DESTINATION ${CMAKE_BINARY_DIR})
    
    message(STATUS "")
    message(STATUS "Visualization executable 'test_viz' will be built")
    message(STATUS "")
endif()

# ============================================================================
# Print Build Summary
# ============================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "Project Configuration")
message(STATUS "========================================")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "Targets:")
message(STATUS "  - geom_core (library with loader.cu)")
message(STATUS "  - test_sdf (CUDA SDF tests)")
if(BUILD_VISUALIZER)
    message(STATUS "  - viz_lib (visualization library)")
    message(STATUS "  - test_viz (OpenGL visualization)")
else()
    message(STATUS "  - test_viz (DISABLED - missing OpenGL dependencies)")
endif()
message(STATUS "========================================")
message(STATUS "")