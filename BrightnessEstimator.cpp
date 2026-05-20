#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <string>
#include "BrightnessEstimator.hpp"
#include <unordered_map>

double calculateBrightness(const cv::Mat& frame) {
    cv::Mat hsv; 
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV); 
    std::vector<cv::Mat> channels; 
    cv::split(hsv, channels); 

    return cv::mean(channels[2])[0];
}

bool isDark(double brightness, const std::unordered_map<std::string, double>& thresholds) {
    if (brightness < thresholds.at("Night")) {
        return true;
    }
    return false;
}

// & -> modifies original variable
void annotateVideo(const std::string& inputPath, const std::string& outputPath, const BrightnessThresholds& thresholds) {
    cv::VideoCapture cap(inputPath); 
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << inputPath; 
        return; 
    }
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)); 
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)); 
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS)); 

    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G'); 
    cv::VideoWriter out (outputPath, fourcc, fps, cv::Size(frameWidth, frameHeight));
    if (!out.isOpened()) {
        std::cerr << "Could not write out video" << std::endl; 
        return; 
    }

    std::map<std::string, int> frameCounts = {{"Night", 0}};
    int frameIndex = 0; 

    while (cap.isOpened()) {
        cv:: Mat frame; 
        bool ret = cap.read(frame); 
        if (!ret || frame.empty()) {
            break;
        }

        double brightness = calculateBrightness(frame); 
        // std::string classification = classifyFrame(brightness, thresholds); 
        frameCounts["Night"]++; 

        std::string text = "Frame: " + std::to_string(frameIndex) + ", Time"; 
        cv::putText(frame, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2); 
        out.write(frame); 
        frameIndex++; 


    }
    cap.release();
    out.release();

    int totalFrames = frameCounts["Night"]; 

    if (totalFrames == 0) {
        std::cout << "No frames classified as night." << std::endl;
        return;
    }

    double nightPercentage = static_cast<double>(frameCounts["Night"]) / totalFrames * 100.0; 
    std::cout << "Video saved to" << outputPath << std::endl; 
}