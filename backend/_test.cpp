#include "hailo/hailort.hpp"
#include <iostream> 
#include <vector>


using namespace hailort; 

int main() {
    const std::string modelPath = "/usr/share/hailo-models/yolov8s_h8l.hef";
    const std::string bodyFace = "/usr/share/hailo-models/yolov5s_personface_h8l.hef";
    const std::string imagePath = "/home/sv/Developer/camera/backend/test.jpg";


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
    auto configured_model_exp = infer_model->configure();

    

}

