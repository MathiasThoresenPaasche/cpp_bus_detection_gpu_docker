cmake_minimum_required(VERSION 3.22)
project(MyProject)

# Compiler
set(CMAKE_CUDA_COMPILER nvcc)

# Flags
set(CMAKE_CUDA_FLAGS "-std=c++17")

# Directories
set(CUDA_INCLUDE /usr/local/cuda/include)
set(OPENCV_INCLUDE /usr/local/include/opencv4)

# Targets
set(TARGET main)

# Source files
set(SRCS main.cpp)

# Include directories
include_directories(${CUDA_INCLUDE})
include_directories(${OPENCV_INCLUDE})

# Find OpenCV package
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found")
endif()

# Compile rule for .cpp files
add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${OpenCV_LIBS})

# Set CUDA flags
set_target_properties(${TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(${TARGET} PRIVATE ${CMAKE_CUDA_FLAGS})
