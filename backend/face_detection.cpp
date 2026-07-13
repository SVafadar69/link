#include "hailo/hailort.hpp"
#include <iostream> 
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#include <algorithm>

using namespace hailort; 

int main() {

    /// 80x80 grid - high resolution predictions for small faces 
    // 40x40 grid - medium resolution predictions for medium faces 
    // 20x20 grid - low-resolution predictions for large faces 

    // classification tensor, bbox tensor, landmark tensor 
    


    const std::string faceDetection = "/home/sv/Developer/camera/backend/hailo8l_models/scrfd_10g.hef";

    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) {
        std::cerr << "Failed to create VDevice: " << std::endl;
        return -1;
    }
    auto vdevice = vdevice_exp.release();
    auto hef_exp = Hef::create(faceDetection);
    if (!hef_exp) {
        std::cerr << "Failed to create HEF Form" << faceDetection << std::endl; 
        return hef_exp.status();    
    }
    auto hef = hef_exp.release();

    auto configure_params = vdevice->create_configure_params(hef);
    auto network_groups_exp = vdevice->configure(hef, configure_params.value());
    if (!network_groups_exp) {
        std::cerr << "Failed to configure VDevice" << std::endl;
        return network_groups_exp.status();
    }
    auto network_groups = network_groups_exp.release();
    auto network_group = network_groups[0];

    auto input_vstreams_params = network_group->make_input_vstream_params(
        false, 
        HAILO_FORMAT_TYPE_UINT8,
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
        HAILO_DEFAULT_VSTREAM_QUEUE_SIZE,
        ""
    );
    auto output_vstreams_params = network_group->make_output_vstream_params(
        false, 
        HAILO_FORMAT_TYPE_FLOAT32,
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
        HAILO_DEFAULT_VSTREAM_QUEUE_SIZE,
        ""
    );
  

    
    auto input_vstreams_exp = VStreamsBuilder::create_input_vstreams(*network_group, input_vstreams_params.value());
    if (!input_vstreams_exp) return input_vstreams_exp.status();
    auto input_vstreams = input_vstreams_exp.release();
    auto output_vstreams_exp = VStreamsBuilder::create_output_vstreams(*network_group, output_vstreams_params.value());
    if (!output_vstreams_exp) return output_vstreams_exp.status();
    auto output_vstreams = output_vstreams_exp.release();

    cv::Mat img = cv::imread("/home/sv/Developer/camera/backend/download_audio.jpg");
    if (img.empty()) {
        std::cerr << "Failed to load image at " << std::endl;
        return HAILO_INVALID_ARGUMENT;
    }
    auto input_shape = input_vstreams[0].get_info().shape;
    int height = input_shape.height; 
    int width = input_shape.width; 

    cv:: Mat resized_img; 
    cv::cvtColor(img, resized_img, cv::COLOR_BGR2RGB);
    cv::resize(resized_img, resized_img, cv::Size(width, height));

    std::cout << "Pushing tensor of shape [" << height << ", " << width << ", 3] to the accelerator" << std::endl; 


    auto write_status = input_vstreams[0].write(MemoryView(resized_img.data, resized_img.total() * resized_img.elemSize()));
    if (write_status != HAILO_SUCCESS) {
        std::cerr << "Failed to write input tensor to the accelerator" << std::endl;
        return write_status;
    }

    std::cout << "\n--- Inference Results ---" << std::endl; 
    for (auto& output_vstream : output_vstreams) {
        auto output_shape = output_vstream.get_info().shape; 
        size_t output_size = output_vstream.get_frame_size();

        std::vector<float> output_data(output_size / sizeof(float)); 

        auto read_status = output_vstream.read(MemoryView(output_data.data(), output_size));
        if (read_status != HAILO_SUCCESS) {
            std::cerr << "Failed reading from output vstream" << std::endl;
            return read_status;
        }

        std::cout << "\nOutput Layer: " << output_vstream.name() << std::endl;
        std::cout << "Shape: [" << output_shape.features << ", " << output_shape.height << ", " << output_shape.width << "]" << std::endl;
        
        // Print a flattened sample of the raw output
        int sample_size = std::min(10, static_cast<int>(output_data.size()));
        std::cout << "Sample values (first " << sample_size << "): ";
        for (int i = 0; i < sample_size; ++i) {
            std::cout << output_data[i] << " ";
        }
        std::cout << std::endl;
    }

    return 0;

}

