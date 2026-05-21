#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <string>

#include "BrightnessEstimator.hpp"

int main() {
    BrightnessThresholds thresholds;
    thresholds.Night = 0;

    const int MAX_STREAMS = 6;

    std::vector<cv::VideoCapture> caps;
    std::vector<int> camIds;

    for (int i = 0; i < 4 && static_cast<int>(caps.size()) < MAX_STREAMS; ++i) {
        cv::VideoCapture cap(i, cv::CAP_ANY);

        if (cap.isOpened() && cap.grab()) {
            std::cout << "Camera index " << i << " opened successfully\n";

            caps.push_back(std::move(cap));
            camIds.push_back(i);
        }
        else {
            std::cout << "Camera index " << i << " not available\n";
        }
    }

    while (static_cast<int>(caps.size()) < MAX_STREAMS) {
        caps.emplace_back();
        camIds.push_back(-1);
    }

    cv::Mat placeholder(
        480,
        640,
        CV_8UC3,
        cv::Scalar(0, 0, 255)
    );

    while (true) {
        for (int i = 0; i < MAX_STREAMS; ++i) {
            std::string title = "Slot " + std::to_string(i);

            cv::Mat frame;

            if (camIds[i] != -1 && caps[i].isOpened()) {
                caps[i].read(frame);
				if (camIds[i] == 2) {
					cv::imshow(title, frame);
				}
				double frame_brightness = calculateBrightness(frame);
				std::cout << "Frame Brightness" << std::to_string(frame_brightness) << std::endl;
				//bool dark = isDark(frame_brightness, thresholds);
				//std::cout << "Is Dark: " << std::to_string(dark) << std::endl;
			}

            
        }

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    for (auto& cap : caps) {
        if (cap.isOpened()) {
            cap.release();
        }
    }

    cv::destroyAllWindows();

    return 0;
}