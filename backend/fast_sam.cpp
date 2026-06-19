#include <opencv2/opencv.hpp>
#include <iostream>
#include "hailo/hailort.hpp"
#include <iostream> 
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>

using namespace hailort; 
// Helper to check Hailo API status
#define REQUIRE_SUCCESS(status, error_msg) \
    do { \
        if ((status) != HAILO_SUCCESS) { \
            std::cerr << (error_msg) << " Status code: " << (status) << "\n"; \
            return -1; \
        } \
    } while(0)

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include "hailo/hailort.hpp"

using namespace hailort;

int main()
{
    std::string hef_path = "fast_sam.hef";

    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) {
        std::cerr << "Failed to create Hailo VDevice\n";
        return -1;
    }
    std::unique_ptr<VDevice> vdevice = vdevice_exp.release();

    auto infer_model_exp = vdevice->create_infer_model(hef_path);
    if (!infer_model_exp) {
        std::cerr << "Failed to create infer model from path. Status: " << infer_model_exp.status() << "\n";
        return -1;
    }
    auto infer_model = infer_model_exp.release();

    auto configured_model_exp = infer_model.configure();
    if (!configured_model_exp) {
        std::cerr << "Failed to configure infer model. Status: " << configured_model_exp.status() << "\n";
        return -1;
    }
    auto configured_model = configured_model_exp.release();

    auto bindings_exp = configured_model.create_bindings();
    if (!bindings_exp) {
        std::cerr << "Failed to create model bindings\n";
        return -1;
    }
    auto bindings = bindings_exp.release();

    auto input_names = infer_model.get_input_names();
    std::string input_name = input_names[0];
    auto input_shape = infer_model.input(input_name)->get_shape(); // NHWC
    int model_w = input_shape.width;
    int model_h = input_shape.height;

    std::cout << "Model Expects Input Size: " << model_w << "x" << model_h << "\n";

    auto output_names = infer_model.get_output_names();
    std::vector<std::vector<uint8_t>> output_buffers;
    for (const auto& name : output_names) {
        size_t out_size = infer_model.output(name)->get_frame_size();
        output_buffers.emplace_back(out_size);
        bindings.output(name)->set_buffer(MemoryView(output_buffers.back().data(), out_size));
    }

    int camera_index = 0;
    cv::VideoCapture cap(camera_index, cv::CAP_ANY);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera " << camera_index << "\n";
        return -1;
    }

    cv::Mat frame, rotatedFrame, resizedFrame, rgbFrame;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: empty frame\n";
            break;
        }

        cv::rotate(frame, rotatedFrame, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::resize(rotatedFrame, resizedFrame, cv::Size(model_w, model_h));
        cv::cvtColor(resizedFrame, rgbFrame, cv::COLOR_BGR2RGB);

        size_t input_size = infer_model.input(input_name)->get_frame_size();
        bindings.input(input_name)->set_buffer(MemoryView(rgbFrame.data, input_size));

        auto status = configured_model.run(bindings, std::chrono::milliseconds(1000));
        if (status != HAILO_SUCCESS) {
            std::cerr << "Inference execution failed! Status: " << status << "\n";
            break;
        }

        // FastSAM output tensors are now available in output_buffers
        cv::imshow("Camera Feed", rotatedFrame);

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}