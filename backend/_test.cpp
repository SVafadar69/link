#include "hailo/hailort.hpp"
#include <iostream> 
#include <vector>
#include <opencv2/opencv.hpp>

using namespace hailort; 

int main() {
    const std::string modelPath = "/usr/share/hailo-models/yolov8s_h8l.hef";
    const std::string bodyFace = "/usr/share/hailo-models/yolov5s_personface_h8l.hef";
    const std::string imagePath = "/home/sv/Developer/camera/backend/download_audio.jpg";


    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) {
        std::cerr << "Failed to create VDevice: " << std::endl;
        return -1;
    }
    auto vdevice = vdevice_exp.release();

    auto infer_model_exp = vdevice->create_infer_model(modelPath);
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

    cv::resize(img, img, cv::Size(640, 640));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    auto input_name = infer_model->get_input_names()[0];
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
        int max_boxes = output_size / (6 * sizeof(float));
        for (int i = 0; i < max_boxes; ++i) {
            float ymin = results[i * 6 + 0];
            float xmin = results[i * 6 + 1];
            float ymax = results[i * 6 + 2];
            float xmas = results[i * 6 + 3];
            float score = results[i * 6 + 4];
            int class_id = static_cast<int>(results[i * 6 + 5]);

            if (score > 0.5f) {
                int x1 = static_cast<int>(xmin * 640);
                int y1 = static_cast<int>(ymin * 640);
                int x2 = static_cast<int>(xmas * 640);
                int y2 = static_cast<int>(ymax * 640);

                cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
                std::string label = "Class " + std::to_string(class_id) + " (" + std::to_string(score).substr(0, 4) + ")";
                cv::putText(img, label, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

            }
        }
        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
        cv::imwrite("inference_result.jpg", img); 
        
    } else {
        std::cerr << "Inference failed with status: " << status << std::endl;
        return -1;
    }
    return 0;

}

