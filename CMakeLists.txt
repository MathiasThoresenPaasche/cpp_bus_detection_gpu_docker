
# cmake_minimum_required(VERSION 3.22)
# project(tracy123)

# # Set C++ standard to C++17
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # Enable optimization with -O2 and use native CPU architecture
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native")

# # Define TRACY_ENABLE for the whole project
# add_compile_definitions(TRACY_ENABLE)
# # Find and link libraries (libpthread and libdl)
# find_package(Threads REQUIRED)
# find_library(LIB_DL dl)
# if (NOT LIB_DL)
#     message(FATAL_ERROR "libdl not found")
# endif ()

# # Tracy integration via CMake FetchContent
# include(FetchContent)
# FetchContent_Declare(
#     tracy
#     GIT_REPOSITORY https://github.com/wolfpld/tracy.git
#     GIT_TAG v0.10
#     GIT_SHALLOW TRUE
#     GIT_PROGRESS TRUE
# )
# FetchContent_MakeAvailable(tracy)

# # Add your own project files
# add_executable(main main.cpp)

# # Link Tracy to your executable
# target_link_libraries(main PUBLIC TracyClient ${CMAKE_DL_LIBS} Threads::Threads)

cmake_minimum_required(VERSION 3.22)
project(MyProject)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# Find OpenCV package
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found")
endif()

# Include directories
include_directories(${CUDA_INCLUDE})
include_directories(${OPENCV_INCLUDE})

# Add Tracy as a dependency using FetchContent
include(FetchContent)
FetchContent_Declare(
    tracy
    GIT_REPOSITORY https://github.com/wolfpld/tracy.git
    GIT_TAG v0.10
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(tracy)

# Add executable
add_executable(${TARGET} ${SRCS})

# Link OpenCV and Tracy to the executable
target_link_libraries(${TARGET} PUBLIC ${OpenCV_LIBS} TracyClient ${CMAKE_DL_LIBS} Threads::Threads)


