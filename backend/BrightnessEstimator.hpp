#pragma once 
#include <string>

struct BrightnessThresholds {
    double Night;
};
bool isDark(double brightness, const std::unordered_map<std::string, double>& thresholds);
double calculateBrightness(const cv::Mat& frame);
void annotateVideo(const std::string& inputPath, const std::string& outputPath, const BrightnessThresholds& thresholds);
// string classifyFrame(double brightness, const)