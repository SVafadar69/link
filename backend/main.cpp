#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include "BrightnessEstimator.hpp"

std::string getEnvValue(const std::string& key) {
    const char* val = std::getenv(key.c_str());
    if (val != nullptr) {
        return std::string(val);
    }
    return std::string();
}

void loadDotenv(const std::string& filename) {
    std::ifstream file(filename); // open file

    if (!file.is_open()) {
        std::cerr << "Error: Could not open .env file: " << filename << "\n"; 
        return;
    }
    std::string line; 

    // loop over each line
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        size_t equalPos = line.find('=');
        // npos -> not found
        if (equalPos == std::string::npos) {
            continue;
        }
        std::string key = line.substr(0, equalPos); // gets everything before it 
        std::string value = line.substr(equalPos + 1); // gets everything after it
        
        if (!value.empty() && value.front() == '"' && value.back() == '"') {
            value = value.substr(1, value.size() - 2);
        } 
    }
}

setenv(key.c_str(), value.c_str(), 1); 

int main() {
    loadDotEnv();
    std::string streamKey = getEnvValue("cloudfare_api_key");
    if (streamKey.empty()) {
        std::cerr << "Error: Could not retrieve key" < std::endl;
        return -1;
    }
    std::string streamUrl = getEnvValue("cloudfare_rtmps") + "/" + streamKey;
    const int camera_index = 0;
    cv::VideoCapture cap(camera_index, cv::CAP_ANY);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 30);

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

    if (fps <= 0) {
        fps = 30; 
    }
    std::ostringstream ffmpegCmd; 

    ffmpegCmd
    << "ffmpeg -loglevel warning "
    << "-f rawvideo "
    << "-pix_fmt bgr24 "
    << "-s " << width << "x" << height << " "
    << "-r " << fps << " "
    << "-i pipe:0 "
    << "-vcodec libx264 "
    << "-pix_fmt yuv420p "
    << "-preset veryfast "
    << "-tune zerolatency "
    << "-profile:v main "
    << "-level 3.1 "
    << "-b:v 4000k "
    << "-maxrate 4000k "
    << "-bufsize 8000k "
    << "-g " << fps << " "
    << "-keyint_min " << fps << " "
    << "-sc_threshold 0 "
    << "-bf 0 "
    << "-f flv "
    << "\"" << rtmpsUrl << "\"";

    FILE* ffmpegPipe = popen(ffmpegCmd.str().c_str(), "w"); 
    if (!ffmpeg) {
        std::cerr << "[ERROR] Failed to start ffmpeg process.\n"; 
        cap.release();
        return 1; 
    }


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
        size_t frameSize = frame.total() * frame.elemSize();
        size_t written = fwrite(flippedFrame.data, 1, frameSize, ffmpeg); 
        if (written != frameSize) {
            std::cerr << "[ERROR] Failed to write full frame"; 
            break;
        }

        cv::imshow("Camera Feed", flippedFrame);
        char key = static_cast<char>(cv::waitKey(1));
        if (key == 'q') {
            break;
        }
    }
    cap.release();
    fflush(ffmpeg); 
    pclose(ffmpeg);
    cv::destroyAllWindows();
    return 0;
}
