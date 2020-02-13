# photometric_stereo

Photometric stereo is a technique in computer vision for estimating the surface normals of objects by observing that object under different lighting conditions. This repo is the implementation of this technique.

Dependencies can be installed using the script `install_dependencies.sh`: 

- OpenCV
- Eigen3
- YAML-CPP

In order to build the software, run the following commands:

``` 
mkdir build
cd build
cmake ..
make 
```

There are 3 executable files : `src/main_gray_data.cpp`, `src/main_color_data.cpp`, `src/face_yale.cpp`, that run photometric stereo on different data types: `gray data`, `color data` and `yale face data`. The executable files are located in the `build` folder.
The variables can be change in the `config.yaml` file.
