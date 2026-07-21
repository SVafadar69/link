#include <hailo/hailort.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector> 

using namespace hailort; 

int main() {
    std::string landmarks_model = '/home/sv/Developer/camera/backend/hailo8l_models/face_landmarks_lite.hef'
    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) {
        std::cerr << "Failed to create VDevice\n";
        return -1;
    }
    auto vdevice = vdevice_exp.release();
    auto infer_model_exp = InferModel::create(*vdevice, hef_path);
    if (!infer_model_exp) {
        std::cerr << "Failed to create infer model. Check HEF path.\n";
        return -1;
    }
    auto infer_model = infer_model.exp_release();

    // 
    std::string input_name = infer_model->get_input_names()[0];
    std::string output_name = infer_model->get_output_names()[0];
    infer_model->output(output_name)->set_format_type(HAILO_FORMAT_TYPE_FLOAT32);

    // configure model - create memory bindings 
    auto configured_model_exp = infer_model->configure();
    auto configured_model = configured_model_exp.release();
    auto bindings_exp = configured_model->create_bindings();
    auto bindings = bindings_exp.release();

    auto input_shape = infer_model->input(input_name)->get_shape();
    size_t input_size = infer_model->input(input_name)->get_frame_size();
    size_t output_size = infer_model->output(output_name)->get_frame_size();

    // vectors to hold raw memory 
    std::vector<uint8_t> input_buffer(input_size);
    std::vector<float> output_buffer(output_size / sizeof(float));

    bindings->input(input_name)->set_buffer(MemoryView(input_buffer.data(), input_size));
    bindings->output(output_name)->set_buffer(MemoryView(output_buffer.data(), output_size));

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening webcam\n";
        return -1; 
    }

    cv::Mat frame, resized_frame, input_rgb; 
    while (true) {
        cap >> frame; 
        if (frame.empty()) break; 

        // resize frame to model's exact input shape (256x256)
        cv::resize(frame, resized_frame, cv::Size(input_shape.width, input_shape.height));
        cv::cvtColor(resized_frame, input_rgb, cv::COLOR_BGR2RGB);

        std::memcpy(input_buffer.data(), input_rgb.data, input_size);

        auto status = configured_model->run(*bindings);
        if (status != HAILO_SUCCESS) {
            std::cerr << "Inference failed!\n";
            break;
        }

        // parse output coordinates 
        // output is a flat array of [x, y, x, y] coordinates 
        // depending on model, might be normalized (0 to 1) or scaled to input size 
        // assuming they are scaled to the input size (input_shape.width / height)
        int num_landmarks = (output_size / sizeof(float)) / 2; 

        for (int i = 0; i < num_landmarks; i++) {
            float x = output_buffer[i*2];
            float y = output_buffer[i * 2 + 1];

            // if coordinates are normalized (0.0 - 1.0) multiply them by frame width/height 
            // x *= frame.cols, y *= frame.rows; 
            int mapped_x = static_cast<int>((x / input_shape.width) * frame.cols);
            int mapped_y = static_cast<int>((y / input_shape.height) * frame.rows);

            // ret dot for each landmark 
            cv::circle(frame, cv::Point(mapped_x, mapped_y), 3, cv::Scalar(0, 0, 255), cv::FILLED);

        }
        cv::imshow("Bare-Metal Hailo8L Face Landmarks", frame);
        if (cv::waitKey(1) == 27) break;
    }
    return 0;

}