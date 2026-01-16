# PaGeRo
This is a parallel computing geometry engine for robot simulations that leverages fast cuda kernels for collision queries and exposes these kernels to be used for planning and obstacle avoidance

## Geometry
Currently this package supports only objects made with Primitives Shapes (sphere, box and cylinder).

The queries regarding the scene geometry are purely distance based giving the signed distance value between two objects or the distance between a point and an object

## Visualization
I have created a simplex graphics pipeline using OPENGL for visulizing the geometry with proper mouse controls. An example visualization can viewed by following the given instructions and running the test case

```bash
git clone https://github.com/aayush-rath/PaGeRo.git

cd PaGeRo

mkdir build
cd build
cmake ..

make -j($nproc)
./test_viz
```

## Requirements
You should have nvcc installed. Having a GPU is not a necessary requirment but is required for making multiple geometric queries in parallel threads

## On Going
This package is still at a very nascent stage and can be improved:

- [ ] Generalizing it for more complex geometry through the use of Bounding Volume Hierarchy and the ability to import mesh files for robots and objects

- [ ] Adding textures in the fragment shaders for visualization for creating objects with better looks

- [ ] Implementing forward and inverse kinematics and also allow moving obstacles in the scene


## P.S.
If you want to use this for your own project or are facing issues or want to improve the current build, feel free to reach out to me via mail

Chalo! Tata Bye bye!