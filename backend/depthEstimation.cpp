#include "hailo/hailort.hpp"
#include <iostream> 
#include <vector>
#include <opencv2/opencv.hpp>

using namespace hailort; 

int main() {
    const std::string depthEstimation = "/home/sv/Developer/camera/backend/hailo8l_models/depth_anything_v2_vits.hef";
    const std::string imagePath = "/home/sv/Developer/camera/backend/download_audio.jpg";


    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) {
        std::cerr << "Failed to create VDevice: " << std::endl;
        return -1;
    }
    auto vdevice = vdevice_exp.release();

    auto infer_model_exp = vdevice->create_infer_model(depthEstimation);
    if (!infer_model_exp) {
        std::cerr << "Error m8" << std::endl;
        return -1; 
    }

    auto infer_model = infer_model_exp.release();
    // dereference infer_model and call function 
    auto configured_model_exp = infer_model->configure();

    if (!configured_model_exp) {
        std::cerr << "Error m8" << std::endl;
        return -1; 
    }

    auto configured_model = configured_model_exp.release();

    auto bindings_exp = configured_model.create_bindings();
    auto bindings = bindings_exp.release();

    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "Failed to load image at " << imagePath << std::endl;
        return -1;
    }

    

    auto input_name = infer_model->get_input_names()[0];
    auto input_shape = infer_model->input(input_name)->shape();
    cv::resize(img, img, cv::Size(input_shape.width, input_shape.height));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    //points to img.data in RAM; 
    bindings.input(input_name)->set_buffer(MemoryView(img.data, img.total()* img.elemSize()));
    
    // name of output layer 
    auto output_name = infer_model->get_output_names()[0];
    // how many bytes of data chip will spit out
    size_t output_size = infer_model->output(output_name)->get_frame_size();
    std::vector<uint8_t> output_buffer(output_size);
    bindings.output(output_name)->set_buffer(MemoryView(output_buffer.data(), output_size));

    // run inference
    const std::chrono::milliseconds time = std::chrono::milliseconds(1000);
    auto status = configured_model.run(bindings, time);
    if (status == HAILO_SUCCESS) {

        float* results = (float*)output_buffer.data(); 

        auto output_shape = infer_model->output(output_name)->shape();
        int out_height = output_shape.height;
        int out_width = output_shape.width;

        cv::Mat depth_raw(out_height, out_width, CV_32FC1, results);
        cv::Mat depth_normalized;
        cv::normalize(depth_raw, depth_normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);

        cv::Mat depth_colormap; 
        cv::applyColorMap(depth_normalized, depth_colormap, cv::COLORMAP_INFERNO);

        cv::imwrite("depth_output.jpg", depth_colormap);
        std::cout << "Depth Estimation Successful" << std::endl;
    }
        
    else {
        std::cerr << "Inference failed with status: " << status << std::endl;
        return -1;
    }
    return 0;

}

