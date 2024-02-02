FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    python3-dev \
    python3-numpy \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libdc1394-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y cuda-toolkit-12-2

# Clone OpenCV and opencv_contrib repositories
RUN git clone --depth 1 --branch  4.8.0 https://github.com/opencv/opencv.git /opencv \
    && git clone --depth 1 --branch  4.8.0 https://github.com/opencv/opencv_contrib.git /opencv_contrib \ 
    && mkdir -p /opencv/build \
    && cd /opencv/build \
    && cmake \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
        -D WITH_CUDA=ON \
        -D ENABLE_FAST_MATH=1 \
        -D CUDA_FAST_MATH=1 \
        -D WITH_CUBLAS=1 \
        -D WITH_TBB=ON \
        -D WITH_V4L=ON \
        -D WITH_GTK=ON \
        -D WITH_OPENGL=ON \
        -D WITH_LIBV4L=ON \
        -D BUILD_opencv_cudacodec=ON \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_python3=ON \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D INSTALL_C_EXAMPLES=OFF \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D OPENCV_ENABLE_NONFREE=OFF \  
        .. \
    && make -j $(nproc) \
    && make install \
    && rm -rf /opencv /opencv_contrib

# Set OpenCV environment variables
ENV OPENCV_VERSION=" 4.8.0"

# Set working directory
WORKDIR /workspace

# Set entrypoint
ENTRYPOINT ["/bin/bash"]