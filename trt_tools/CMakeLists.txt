cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 14)
add_definitions(-O0 -g)

include_directories(/data1/virgo_3rdparty/TensorRT-8.4.1.5/include)
include_directories(/data1/zhn/code/eigen-git-mirror/install/include/eigen3)

link_directories(/data1/virgo_3rdparty/TensorRT-8.4.1.5/lib)

add_executable(testinter test.cpp trtInterface.cpp trtInterface.h error_recorder.h logger.h logger.cpp logging.h common.h superp.h superp.cpp buffers.h half.h)
target_link_libraries(testinter
  nvinfer
  nvonnxparser
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBRARIES}
)