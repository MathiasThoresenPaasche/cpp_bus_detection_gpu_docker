#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>

int main() {
    std::cout << "Hello from bus_detection" << std::endl;
    // Print CUDA version
    int cudaVersion;
    cudaRuntimeGetVersion(&cudaVersion);
    std::cout << "CUDA Version: " << cudaVersion << std::endl;
    // std::cout << "CUDA Version: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;


    // List CUDA devices
    // cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());
    cv::VideoCapture cap(4);
    
    // Check if the webcam device is opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open the webcam device." << std::endl;
        return -1;
    }
    // Create a window to display the webcam feed
    cv::namedWindow("Webcam Feed", cv::WINDOW_NORMAL);
    while (true) {
        cv::Mat frame;
        cap >> frame;
        // Check if the frame is empty
        if (frame.empty()) {
            std::cerr << "Error: Unable to capture frame." << std::endl;
            break;
        }
        cv::imshow("Webcam Feed", frame);
        // Check for 'Esc' key press to exit
        if (cv::waitKey(1) == 27)
            break;
    }
    // Release the webcam device and close the window
    cap.release();
    cv::destroyAllWindows();

    // Test OpenCV
    cv::Mat mat(2, 2, CV_8UC1);
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    std::cout << "Matrix size: " << mat.size() << std::endl;

    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
            prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
    return 0;
}

// #include <iostream>
// #include </usr/local/cuda-12.2/targets/x86_64-linux/include/cuda_runtime.h>

// int main() {
//     std::cout << "Hello from bus_detection" << std::endl;
//     // Print CUDA version
//     int cudaVersion;
//     cudaRuntimeGetVersion(&cudaVersion);
//     std::cout << "CUDA Version: " << cudaVersion << std::endl;
//     //  hello
//     return 0;
// }

/*
g++ -o test_opencv_cuda test_opencv_cuda.cpp \
    -I/usr/local/cuda-12.2/include
    -L/usr/local/cuda/lib64 \ # Add CUDA library path if necessary
       
       
       
       
       
       
        -I/usr/local/include/opencv4 \

    -lopencv_core   \
     -lopencv_imgcodecs -lopencv_imgproc \
    -lopencv_highgui -lopencv_videoio -lopencv_video \
    -lopencv_objdetect -lopencv_features2d -lopencv_calib3d


*/
