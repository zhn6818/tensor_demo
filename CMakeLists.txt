cmake_minimum_required(VERSION 3.20)

project(resnet)

add_definitions(-std=c++11)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

add_definitions(-O0 -g)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/data1/virgo_3rdparty/TensorRT-8.4.1.5/include/)
link_directories(/data1/virgo_3rdparty/TensorRT-8.4.1.5/lib/)

add_executable(resnet18 ${PROJECT_SOURCE_DIR}/resnet18.cpp)
target_link_libraries(resnet18 nvinfer libnvinfer.so nvparsers nvinfer_plugin nvonnxparser)
target_link_libraries(resnet18 cudart )

add_executable(testOnnx ${PROJECT_SOURCE_DIR}/test_onnx.cpp )
target_link_libraries(testOnnx nvinfer libnvinfer.so nvparsers nvinfer_plugin nvonnxparser)
target_link_libraries(testOnnx cudart )


add_executable(superpoint ${PROJECT_SOURCE_DIR}/superpoint.cpp )
target_link_libraries(superpoint nvinfer libnvinfer.so nvparsers nvinfer_plugin nvonnxparser)
target_link_libraries(superpoint cudart )

add_subdirectory(trt_tools)
# message("~~~~~~~~~~~~~")
# message(${CUDA_LIBRARIES})
# message("~~~~~~~~~~~~~")


# add_executable(resnet34 ${PROJECT_SOURCE_DIR}/resnet34.cpp)
# target_link_libraries(resnet34 nvinfer)
# target_link_libraries(resnet34 cudart)

# add_executable(resnet50 ${PROJECT_SOURCE_DIR}/resnet50.cpp)
# target_link_libraries(resnet50 nvinfer)
# target_link_libraries(resnet50 cudart)

# add_executable(resnext50 ${PROJECT_SOURCE_DIR}/resnext50_32x4d.cpp)
# target_link_libraries(resnext50 nvinfer)
# target_link_libraries(resnext50 cudart)

# add_executable(wideresnet50 ${PROJECT_SOURCE_DIR}/wideresnet50.cpp)
# target_link_libraries(wideresnet50 nvinfer)
# target_link_libraries(wideresnet50 cudart)

# add_definitions(-O0 -pthread)

