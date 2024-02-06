# ------ Configure and build Docker image ---------

# Changes that has to be made to the Dockerfile before you run docker build:
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 - correct cuda version?
RUN apt-get update && apt-get install -y cuda-toolkit-12-1 - correct cuda version?
opencv - correct opencv version?
       - corrct flags?
       - correct arcitechture version on line "-D CUDA_ARCH_BIN=8.6". called "Compute Capability" when serached for

And most importantly, does your GPU, CUDA vesrion, CUDNN version , CUDA-Toolkit version, OpenCV version and OpenCV architecture version work togheter?


# Build docker image in folder from Dockerfile in folder:
sudo docker build -t bus_detection .

# ------ Run Docker image ---------
# Run docker image with connection to computer folder and all GPUs

sudo docker run --gpus all -v "$(pwd)":/workspace -it bus_detection:latest

# Run docker image with connection to computer folder, all GPUs and graphics output. Run from folder with main folder in porject. Note: --device /dev/video4:/dev/video4  has to be changed to match the device of your camera input. 
sudo docker run  \
--device /dev/video4:/dev/video4 \
--gpus all -v "$(pwd)":/workspace \
-it \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
bus_detection:latest

# ------ Compile code ---------------
# Use Makefile

# !OLD! Compile code with nvidia compiler for use of CUDA
nvcc -o test_opencv_cuda test_opencv_cuda.cpp

# !OLD! Compile code with nvidia compiler for use of CUDA with opencv4
nvcc -o test_opencv_cuda test_opencv_cuda.cpp -I/usr/local/include/opencv4


# To view graphics - Lazy and unsafe method. Run on host before starting docker container
xhost +local:root # for the lazy and reckless
# To turn it off afterwards
xhost -local:root





# Disregard! For other docker image with yolo
sudo docker run  \
--device /dev/video4:/dev/video4 \
--gpus all -v "$(pwd)":/workspace \
--ipc=host --ulimit memlock=-1 --ulimit stack=67108864
-it \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
ultralytics/yolov5:v7.0

sudo docker run \
--ipc=host \
--gpus all \
-it -v "$(pwd)":/usr/src/workspace \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
ultralytics/yolov5:v7.0
