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


# To view graphics - Lazy and unsafe method 
xhost +local:root # for the lazy and reckless
# To turn it off afterwards
xhost -local:root



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
