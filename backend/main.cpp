#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include "BrightnessEstimator.hpp"

int main() {
    const int camera_index = 0;
    cv::VideoCapture cap(camera_index, cv::CAP_ANY);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera " << camera_index << "\n";
        return -1;
    }
    cv::Mat frame; 
    cv:: Mat flippedFrame; 

    while (true) {
        cap >> frame; 
        if (frame.empty()) {
            std::cerr << "Error: empty frame"; 
            break;
        }
        cv::rotate(frame, flippedFrame, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::imshow("Camera Feed", flippedFrame);
        char key = static_cast<char>(cv::waitKey(1));
        if (key == 'q') {
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
