#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp> 
#include <stdio.h> 
#include <mutex>
#include <thread>
#include <condition_variable>
#include <vector>
#include <functional>
#include <future>
#include <fstream>
#include <variant>
#include <typeinfo>

std::mutex mtx;
std::condition_variable cond_var;
bool data_ready = false;
bool stop = false;
const std::chrono::milliseconds max_time_gap = static_cast<std::chrono::milliseconds>(1200);
// struct ImagePair{
//     cv::Mat rgb_image;
//     cv::Mat edg_image;
// };
// std::deque<ImagePair> image_buffer;
struct Image{
    cv::Mat rgb_image;
    std::atomic_int processing_counter = 0;
};

Image current_image;

enum CameraEnum{
    camera_pc_built_in = 0,
    camera_pc_ir = 2,
    camera_logtiech_web = 4 
};

enum Classes {
    person,
    bicycle,
    car,
    motorbike,
    aeroplane,
    bus,
    train,
    truck,
    boat,
    traffic_light,
    fire_hydrant,
    stop_sign,
    parking_meter,
    bench,
    bird,
    cat,
    dog,
    horse,
    sheep,
    cow,
    elephant,
    bear,
    zebra,
    giraffe,
    backpack,
    umbrella,
    handbag,
    tie,
    suitcase,
    frisbee,
    skis,
    snowboard,
    sports_ball,
    kite,
    baseball_bat,
    baseball_glove,
    skateboard,
    surfboard,
    tennis_racket,
    bottle,
    wine_glass,
    cup,
    // fork,
    knife,
    spoon,
    bowl,
    banana,
    apple,
    sandwich,
    orange,
    broccoli,
    carrot,
    hot_dog,
    pizza,
    donut,
    cake,
    chair,
    sofa,
    potted_plant,
    bed,
    dining_table,
    toilet,
    tv_monitor,
    laptop,
    mouse,
    remote,
    keyboard,
    cell_phone,
    microwave,
    oven,
    toaster,
    sink,
    refrigerator,
    book,
    clock_,
    vase,
    scissors,
    teddy_bear,
    hair_dryer,
    toothbrush
};

struct BusElement{
    int box_x;
    int box_y;
    int box_width;
    std::chrono::high_resolution_clock::time_point time_observed;

};
void camera_to_stream(){
    std::cout << "Camera to stream start" << std::endl;
    
    cv::VideoCapture camera(CameraEnum::camera_logtiech_web); 
    if (!camera.isOpened()){
        std::cout << "Could not open camera.";
    }
    cv::Mat image_read;
    while (true){
        camera >> image_read;
        {
            std::lock_guard lock(mtx);
            current_image.rgb_image = image_read;
            data_ready = true;
        }
        cond_var.notify_all();
    }
    camera.release();
}

void rgb_image_stream(){
    std::cout << "rgb_image_stream started" << std::endl;
    cv::namedWindow("Camera feed",cv::WINDOW_NORMAL);

    while (true){
        std::unique_lock<std::mutex> consumer_lock(mtx);
        cond_var.wait(consumer_lock, [] { return data_ready; });
        
        cv::imshow("Camera feed", current_image.rgb_image);
        if (cv::waitKey(1) == 27) { //key 27 = esc
                std::lock_guard lock(mtx);
                stop = true;
                break;
            }
        current_image;
        data_ready = false;

        if (current_image.processing_counter < 2){
            cond_var.wait(consumer_lock);
        }
        else{
            cond_var.notify_one();
            current_image.processing_counter = 0;
        }
    }
}

// void edge_image_stream(){
//     std::cout << "edge_image_stream started" << std::endl;
//     cv::namedWindow("Edge detection feed",cv::WINDOW_NORMAL);
//     cv::Mat edge_image;
//     while (true){
//         std::unique_lock<std::mutex> consumer_lock(mtx);
//         {
//         cond_var.wait(consumer_lock, [] { return data_ready; });
//         sobel_edge(current_image.rgb_image,edge_image);
//         }
//         current_image.processing_counter++;
//         data_ready = false;

//         if (current_image.processing_counter < 2){
//             cond_var.wait(consumer_lock);
//         }
//         else{
//             cond_var.notify_one();
//             current_image.processing_counter = 0;
//         }
//         cv::imshow("Edge detection feed", edge_image);
//         if (cv::waitKey(1) == 27) { //key 27 = esc
//                 std::lock_guard lock(mtx);
//                 stop = true;
//                 break;
//             }
//     }
// }


std::vector<std::string> load_class_list()
{
    std::vector<std::string> class_list;
    std::ifstream ifs("../yolov5-opencv-cpp-python/config_files/classes.txt");
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}

void load_net(cv::dnn::Net &net, bool is_cuda)
{
    auto result = cv::dnn::readNet("yolov5m.onnx");
    if (is_cuda)
    {
        std::cout << "Attempty to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else
    {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.4;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.5;

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

cv::Mat format_yolov5(const cv::Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className) {
    cv::Mat blob;

    auto input_image = format_yolov5(image);
    
    cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    
    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;
    
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {

        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {

            float * classes_scores = data + 5;
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD) {

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }

        }

        data += 85;

    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}

int main(int argc, char **argv)
{
    const cv::Point pt_left(0,195);
    const cv::Point pt_right(639,300);
    // cv::namedWindow("view", cv::WINDOW_NORMAL);
    // cv::Mat view_image= cv::imread("media/camera_view.jpg");
    // cv::line(view_image,pt_left, pt_right, cv::Scalar(0,0,255,3));
    // cv::imshow("view", view_image);
    // cv::waitKey(1);
    std::vector<std::string> class_list = load_class_list();

    cv::Mat frame;
    // cv::VideoCapture capture("/home/mtp/cpp_motion_detection/media/bus_to_oslo_long.mp4");
    cv::VideoCapture capture(CameraEnum::camera_logtiech_web);
    if (!capture.isOpened())
    {
        std::cerr << "Error opening video file\n";
        return -1;
    }

    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;

    cv::dnn::Net net;
    load_net(net, is_cuda);

    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;
    int total_frames = 0;


    // Convert to time in string
    auto system_time = std::chrono::time_point_cast<std::chrono::system_clock::duration>(start);
    std::time_t time_tt = std::chrono::system_clock::to_time_t(system_time);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_tt), "%Y-%m-%d");
    std::string dateString = oss.str();

    // File to store bus observations
    std::string file_name = "bus_times/bus_times_" + dateString;
    std::cout << "filename: " << file_name << std::endl;
    std::ofstream outputFile(file_name, std::ios::app);
    if (!outputFile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1; // Return an error code
    }
    outputFile << "pt_left: " << pt_left << " , " << "pt_right: " << pt_right << std::endl << ";;" << std::endl;

    std::vector<BusElement> bus_observations;

    while (true)
    {   
        try{
            capture.read(frame);
        }catch(const cv::Exception& ex){
            std::cerr << "OpenCV Error: " << ex.what() << std::endl;
            break;
        }catch(const std::exception &ex){
            std::cerr << "Exception: " << ex.what() << std::endl;
            break;
        }catch (...) {
            std::cerr << "Unknown exception occurred." << std::endl;
        }

        
        if (frame.empty())
        {
            std::cout << "End of stream\n";
            break;
        }
        std::vector<Detection> output;
        detect(frame, net, output, class_list);

        frame_count++;
        total_frames++;
        int detections = output.size();

        for (int i = 0; i < detections; ++i)
        {

            auto detection = output[i];
            auto box = detection.box;
            auto classId = detection.class_id;
            if (classId == Classes::bus){
                BusElement bus_elem{box.x, box.y,box.width,std::chrono::high_resolution_clock::now()};
                bus_observations.push_back(bus_elem);
            }
            const auto color = colors[classId % colors.size()];
            cv::rectangle(frame, box, color, 3);

            cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
            cv::putText(frame, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }

        if (frame_count >= 30)
        {
            auto end = std::chrono::high_resolution_clock::now();
            fps = frame_count * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            frame_count = 0;
            start = std::chrono::high_resolution_clock::now();
        }

        if (fps > 0)
        {
            std::ostringstream fps_label;
            fps_label << std::fixed << std::setprecision(2);
            fps_label << "FPS: " << fps;
            std::string fps_label_str = fps_label.str();

            cv::putText(frame, fps_label_str.c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }
        cv::line(frame,pt_left, pt_right, cv::Scalar(0,0,255,3));

        cv::imshow("output", frame);

        if (!bus_observations.empty()){
            auto last_observation = bus_observations.back().time_observed;
            auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(start-last_observation);
            if (time_diff > max_time_gap){
                std::cout << "New bus observation" << std::endl;
                for (const auto& observations : bus_observations) {
                    auto t_c = std::chrono::time_point_cast<std::chrono::system_clock::duration>(observations.time_observed);
                    std::time_t timeT = std::chrono::system_clock::to_time_t(t_c);

                    // Write to the file instead of std::cout
                    outputFile << "X: " << observations.box_x << " , Y: " << observations.box_y << " , "
                    << "bounding_box_width: " <<observations.box_width <<" , "<< std::put_time(std::localtime(&timeT), "%F %T.") << std::endl;
                }
                outputFile << ";;" << std::endl;
                bus_observations = {};
            }
        }
        if (cv::waitKey(1) != -1)
        {
            capture.release();
            std::cout << "finished by user\n";
            break;
        }
    }
    

    std::cout << "Total frames: " << total_frames << "\n";
    // std::cout << "Bus observations:" << std::endl;
    // for (auto observations:bus_observations){
    //     auto t_c = std::chrono::time_point_cast<std::chrono::system_clock::duration>(observations.time_observed);
    //     std::time_t timeT = std::chrono::system_clock::to_time_t(t_c);
    //     std::cout << "X: "<< observations.box_x << " , Y: " << observations.box_y << " , " <<std::put_time(std::localtime(&timeT), "%F %T.\n") << std::flush;
    // }

    if (!bus_observations.empty()){
        for (const auto& observations : bus_observations) {
            auto t_c = std::chrono::time_point_cast<std::chrono::system_clock::duration>(observations.time_observed);
            std::time_t timeT = std::chrono::system_clock::to_time_t(t_c);

            // Write to the file instead of std::cout
            outputFile << "X: " << observations.box_x << " , Y: " << observations.box_y << " , "
            << std::put_time(std::localtime(&timeT), "%F %T.") << std::endl;
        }
        outputFile << ";;" << std::endl;
    }
    outputFile.close();
    return 0;
}