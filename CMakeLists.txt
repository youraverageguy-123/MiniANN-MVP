cmake_minimum_required(VERSION 3.14)
project(MiniANN VERSION 1.3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

enable_testing()

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
  src/visualizer.cpp
  src/experiment.cpp
  src/normalize.cpp
  src/validate.cpp
  src/datasource.cpp
)
target_include_directories(miniann PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(miniann PUBLIC cxx_std_17)
add_library(miniann::miniann ALIAS miniann)
set_target_properties(miniann PROPERTIES VERSION 1.3 SOVERSION 1)

add_executable(xor_demo demos/xor_demo.cpp)
target_link_libraries(xor_demo miniann)

add_executable(and_or_demo demos/and_or_demo.cpp)
target_link_libraries(and_or_demo miniann)

add_executable(iris_demo demos/iris_demo.cpp)
target_link_libraries(iris_demo miniann)

add_executable(compare_demo demos/compare_demo.cpp)
target_link_libraries(compare_demo miniann)

add_executable(playground demos/playground.cpp)
target_link_libraries(playground miniann)

add_executable(gradient_check tests/gradient_check.cpp)
target_link_libraries(gradient_check miniann)

add_executable(test_basic tests/test_basic.cpp)
target_link_libraries(test_basic miniann)

add_executable(test_choices tests/test_choices.cpp)
target_link_libraries(test_choices miniann)

add_executable(test_training tests/test_training.cpp)
target_link_libraries(test_training miniann)

add_executable(test_correctness tests/test_correctness.cpp)
target_link_libraries(test_correctness miniann)

add_executable(test_patterns tests/test_patterns.cpp)
target_link_libraries(test_patterns miniann)

add_test(NAME test_basic COMMAND test_basic)
add_test(NAME test_choices COMMAND test_choices)
add_test(NAME gradient_check COMMAND gradient_check)
add_test(NAME test_training COMMAND test_training)
add_test(NAME test_correctness COMMAND test_correctness)
add_test(NAME test_patterns COMMAND test_patterns)

# Optional native GUI: only built when Qt6 Widgets are available, so plain
# `cmake -B build && cmake --build build` keeps working without them.
# (demos/gui_qt.cpp uses no Q_OBJECT, so no AUTOMOC handling is needed.)
find_package(Qt6 QUIET COMPONENTS Widgets)
if(Qt6Widgets_FOUND)
  add_executable(gui_qt demos/gui_qt.cpp)
  target_link_libraries(gui_qt miniann Qt6::Widgets)
else()
  message(STATUS "Qt6 Widgets not found - skipping gui_qt (install Qt6 to enable the GUI)")
endif()

# ---- Library packaging: install + find_package support ----
# Keeps MiniANN usable as a real library, not just demos-in-a-repo.
include(GNUInstallDirs)
install(TARGETS miniann EXPORT MiniANNTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/miniann DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT MiniANNTargets NAMESPACE miniann::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MiniANN)
