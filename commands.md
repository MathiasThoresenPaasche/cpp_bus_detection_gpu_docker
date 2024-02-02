# Build docker image in folder from Dockerfile in folder:

sudo docker build -t bus_detection .


# Run docker image with connection to computer folder and all GPUs

sudo docker run --gpus all -v "$(pwd)":/workspace -it bus_detection:latest

sudo docker run  \
--device /dev/video4:/dev/video4 \
--gpus all -v "$(pwd)":/workspace \
-it \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
bus_detection:latest


# Compile code with nvidia compiler for use of CUDA

nvcc -o test_opencv_cuda test_opencv_cuda.cpp

# Compile code with nvidia compiler for use of CUDA with opencv4
# See updated Makefile

nvcc -o test_opencv_cuda test_opencv_cuda.cpp -I/usr/local/include/opencv4

