cmake_minimum_required(VERSION 3.27)
project(halide_experiments)

set(CMAKE_CXX_STANDARD 17)

find_package(Halide REQUIRED)

include_directories(include)

file(GLOB_RECURSE LIBS lib/**/*.*)
file(GLOB_RECURSE SOURCES_C "src/*.c")
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "src/*.h")

add_executable(halide_experiments ${SOURCES} ${SOURCES_C} ${HEADERS} ${LIBS} main.cpp)
target_link_libraries(halide_experiments PRIVATE Halide)

add_executable(pixel_differences ${SOURCES} ${SOURCES_C} ${HEADERS} ${LIBS} pixel_differences.cpp)
target_link_libraries(pixel_differences PRIVATE Halide)