cmake_minimum_required(VERSION 3.14)
project(MiniANN VERSION 1.3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

include_directories(include)

add_library(miniann
  src/activation.cpp
  src/neuron.cpp
  src/layer.cpp
  src/network.cpp
  src/loss.cpp
  src/optimizer.cpp
  src/trainer.cpp
  src/dataset.cpp
  src/metrics.cpp
  src/logger.cpp
  src/serializer.cpp
)

add_executable(xor_demo demos/xor_demo.cpp)
target_link_libraries(xor_demo miniann)

add_executable(and_or_demo demos/and_or_demo.cpp)
target_link_libraries(and_or_demo miniann)

add_executable(iris_demo demos/iris_demo.cpp)
target_link_libraries(iris_demo miniann)

add_executable(gradient_check tests/gradient_check.cpp)
target_link_libraries(gradient_check miniann)

add_executable(test_basic tests/test_basic.cpp)
target_link_libraries(test_basic miniann)
