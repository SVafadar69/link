#include "hailo/hailort.hpp"
#include <iostream> 
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#include <algorithm>
#include <array>
#include <map>
#include <numeric> 
#include <cmat> 

using namespace hailort; 


struct OutputTensor {
    std:: string name, 
    int height; 
    int width; 
    int channels; 

    std::vector<float> data;
};

struct FaceDetection {
    cv::Rect2f bbox; 
    float score; 

    std::array<cv::Point2f, 5>landmarks; 
};

struct SCRFDBranch {
    const OutputTensor *scores = nullptr; 
    const OutputTensor *bboxes = nullptr;
    const OutputTensor *landmarks = nullptr;
};

static float tensorAt(const OutputTensor &tensor, int y, int x, int channel) {
    const size_t index = (static_cast<size_t>(y) * tensor.width + x) * tensor.channels + channel; 

    return tensor.data.at(index);
}

static float intersectionOverUnion(const cv::Rect2f &a, const cv::Rect2f &b) {
    const float intersectionLeft = std::max(a.x, b.x);
    const float intersectionTop = std::max(a.y, b.y); 
    const float intersectionRight = std::min(a.x + a.width, b.x + b.width);
    const float intersectionBottom = std::min(a.y + a.height, b.y + b.height);
    const float intersectionWidth = std::max(0.0f, intersectionRight - intersectionLeft); 
    const float intersectionHeight = std::max(0.0f, intersectionBottom - intersectionTop);
    const float intersectionArea = intersectionWidth * intersectionHeight; 
    const float unionArea = a.area() + b.area() - intersectionArea; 

    if (unionArea <= 0.0f) {
        return 0.0f;
    }

    return intersectionArea / unionArea; 
}

static std::vector<FaceDetection> applyNMS(std::vector<FaceDetection> detections, float iouThreshold) {
    std::sort(detections.begin(), detections.end(), [](const FaceDetection &a, const FaceDetection &b) {
        return a.score > b.score; 
    });
    std::vector<bool> suppresed(detections.size(), false);
    std::vector<FaceDetection> kept; 

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppresed[i]) {
            continue;
        }
        kept.push_back(detections[i]);
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppresed[j]) {
                continue; 
            }
            const float iou = intersectionOverUnion(detections[i].bbox, detections[j].bbox);
            if (iou > iouThreshold) {
                suppresed[j] = true; 
            }
        }
    }
    return kept;
}

static std::vector<FaceDetection> postProcessSCRFD(const std::vector<OutputTensor> &outputs, int inputWidth, int inputHeight, 
float scoreThreshold = 0.5f, float nmsThreshold = 0.4f) {
    std::map<int, SCRFDBranch> branches;
    for (const auto &tensor : outputs) {
        if (tensor.width <= 0 || tensor.height <= 0) {
            continue; 
        }
        const int strideX = inputWidth / tensor.width; 
        const int strideY = inputHeight / tensor.height; 
        if (strideX != strideY) {
            std::cerr << "Unexpected non-square stride for " << tensor.name << std::endl; 
            continue; 
        }
        const int stride = strideX; 
        auto &branch = branches[stride]; 

        switch (tensor.channels) {
            case 2: 
                branch.scores = &tensor; 
                break; 
            case 8: 
                branch.bboxes = &tensor; 
                break 
            case 20: 
                branch.landmarks= &tensor; 
                break; 
            default: 
                std::cerr << "Ignoring unexpected SCRFD output " << tensor.name << "with " << tensor.channels << " channels" << std::endl; 
            break; 
        }
    }
    std::vector<FaceDetection> candidates; 

    for (const auto &[stride, branch] : branches) {
        if (branch.scores == nullptr || branch.bboxes == nullptr || branch.landmarks == nullptr) {
            std::cerr << "Incomplete SCRFD branch for stride " << stride << std::endl; 
            continue;
        }

        const int gridHeight = branch.scores->height; 
        const int gridWidth = branch.scores->width; 

        for (int y = 0; y < gridHeight; ++y) {
            for (int x = 0; x < gridWidth; ++x) {
                const float centerX = static_cast<float>(x * stride); 
                const float centerY = static_cast<float>(y * stride); 
                for (int anchor = 0; anchor < 2; ++anchor) {
                    const float score = tensorAt(*branch.scores, y, x, anchor); 
                    if (score < scoreThreshold) {
                        continue; 
                    }

                    const int boxBase = anchor * 4; 
                    const float leftDistance = tensorAt(*branch.boxes, y, x, boxBase + 0); 
                    const float topDistance = tensorAt(*branch.boxes, y, x, boxBase + 1); 
                    const float rightDistance = tensorAt(*branch.boxes, y, x, boxBase + 2); 
                    const float bottomDistance = tensorAt(*branch.boxes, y, x, boxBase + 3); 

                    float x1 = centerX - leftDistance * stride; 
                    float y1 = centerY - topDistance * stride; 
                    float x2 = centerX + rightDistance * stride; 
                    float y2 = centerY + bottomDistance * stride;  

                    x1 = std::clamp(x1, 0.0f, static_cast<float>(inputWidth - 1)); 
                    y1 = std::clamp(y1, 0.0f, static_cast<float>(inputHeight - 1)); 
                    x2 = std::clamp(x2, 0.0f, static_cast<float>(inputWidth - 1)); 
                    y2 = std::clamp(y2, 0.0f, static_cast<float>(inputHeight - 1));

                    if (x2 <= x1 || y2 <= y1) {
                        continue; 
                    }

                    FaceDetection detection; 
                    detection.score = score; 

                    detection.box = cv::Rect2f(x1, y1, x2 - x1, y2 - y1); 

                    const int landmarkBase = anchor * 10; 
                    for (int landmark = 0; landmark < 5; ++landmark) {
                        const float offsetX = tensorAt(*branch.landmarks, y, x, landmarkBase + landmark * 2);
                        const float offsetY = tensorAt(*branch.landmarks, y, x, landmarkBase + landmark * 2 + 1); 
                        detections.landmarks[landmark] = cv::Point2f(centerX + offsetX * stride, centerY + offsetY * stride); 
                    }
                    candidates.push_back(detection);
                    
                }
            }
        }
    }
    return applyNMS(candidates, NMSThreshold);
}

static cv::Mat alignFaceUsingEyes(const cv::Mat &image, cv::Point2f eyeA, cv::Point2f eyeB, int outputWidth = 112, int outputHeight = 112) {
    cv::Point2f leftImageEye = eyeA; 
    cv::Point2f rightImageEye = eyeB; 
    if (leftImageEye.x > rightImageEye.x) {
        std::swap(leftImageEye, rightImageEye);
    }
    const float deltaX = rightImageEye.x - leftImageEye.x; 
    const float deltaY = rightImageEye.y - leftImageEye.y; 
    const float currentEyeDistance = std::sqrt(deltaX * deltaX + deltaY * deltaY); 

    if (currentEyeDistance < 1.0f) {
        return {};
    }

    const float targetLeftEyeX = 
}


int main() {

    /// 80x80 grid - high resolution predictions for small faces 
    // 40x40 grid - medium resolution predictions for medium faces 
    // 20x20 grid - low-resolution predictions for large faces 

    // classification tensor, bbox tensor, landmark tensor 
    // [2, 80, 80] - 2 prediction slots x 1 face score | 2 prediction slots x 4 distances | 2 prediction slots x 5 landmarks x 2 coordinates 
    


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

