# Compiler
NVCC := nvcc

# Flags
NVCCFLAGS := -std=c++17 -lcudart -lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_highgui

# Directories
CUDA_INCLUDE := /usr/local/cuda/include
OPENCV_INCLUDE := /usr/local/include/opencv4

# Targets
TARGET := test_opencv_cuda

# Source files
SRCS := test_opencv_cuda.cpp
# You can add more .cpp or .cu files here if needed

# Object files
OBJS := $(SRCS:.cpp=.o)
# You can add more .cu files here if needed

# Compile rule for .cpp files
%.o: %.cpp
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) -I$(CUDA_INCLUDE) -I$(OPENCV_INCLUDE)

# Compile rule for .cu files
# Uncomment and modify this if you have .cu files
# %.o: %.cu
# 	$(NVCC) $(NVCCFLAGS) -c $< -o $@ -I$(CUDA_INCLUDE) -I$(OPENCV_INCLUDE)

# Link rule
$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $(TARGET) $(NVCCFLAGS)

.PHONY: clean

clean:
	rm -f $(OBJS) $(TARGET)



