cmake_minimum_required (VERSION 3.1)

project (photometric_stereo)

add_compile_options(-std=c++14)
set(CMAKE_BUILD_TYPE "Release")
find_package(OpenMP)
if (OPENMP_FOUND)
    MESSAGE("OpenMP DIR : ${OpenMP_DIR}")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED )

include_directories(
    common/include
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
    )


file(GLOB sources "${PROJECT_SOURCE_DIR}/common/src/*.cpp")
add_library(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} ${Eigen3_INCLUDE_DIRS} ${YAML_CPP_LIBRARIES} )

#1.1 Estimating Albedo and Surface Normal
#Color
add_executable(main_color_data src/main_color_data.cpp )
target_link_libraries(main_color_data ${PROJECT_NAME})
#Gray
add_executable(main_gray_data src/main_gray_data.cpp )
target_link_libraries(main_gray_data ${PROJECT_NAME})

# Test on Yale face dataset
add_executable(face_yale src/face_yale.cpp )
target_link_libraries(face_yale ${PROJECT_NAME})


