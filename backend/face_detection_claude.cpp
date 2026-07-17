#include <hailo/hailort.hpp>
#include <iostream> 
#include <vector> 
#include <array>
#include <cmath> 
#include <string> 
#include <algorithm> 
#include <opencv2/opencv.hpp>

using namespace hailort; 

constexpr float SCORE_THRESH = 0.5f; 
constexpr float IOU_THRESH = 0.4f; 
constexpr bool SCORES_ARE_LOGITS = true; 

struct FaceDetection {
    float score; 
    float x1, y1, x2, y2; 
    std::array<cv::Point2f, 5> landmarks; 
}; 

struct RawOutput {
    std::string name; 
    int features; 
    int height; 
    int width; 
    std::vector<float> data; // NWHC layout: data([y * width + x] * features + c)]

};

inline float sigmoid(float x) {return 1.0f / (1.0f + std::exp(-x)); }

// each anchor point predicts (l, t, r, b) distances to the box edges and
// (dx, dy) offsets for each of the 5 landmarks, all scaled by `stride`

std::vector<FaceDetection> decode_stride(const RawOutput& score_out, const RawOutput& bbox_out, const RawOutput* kps_out, 
int stride, float score_thresh, bool scores_are_logits) {
    std::vector<FaceDetection> dets; 
    const int grid_h = score_out.height; 
    const int grid_w = score_out.width; 
    const int num_anchors = score_out.features; // score channels == num_anchors
    const int bbox_per_anchor = bbox_out.features / num_anchors; // expect 4 
    const int kps_per_anchor = kps_out ? kps_out->features / num_anchors : 0; // expect 10 

    for (int y = 0; y < grid_h; ++y) {
        for (int x = 0; x < grid_w; ++x) {
            for (int a = 0; a < num_anchors; ++a) {
                const int score_idx = (y * grid_w + x) * num_anchors + a; 
                const float raw_score = score_out.data[score_idx]; 
                const float score = scores_are_logits ? sigmoid(raw_score) : raw_score; 
                if (score < score_thresh) continue; 

                const float cx = static_cast<float>(x * stride); 
                const float cy = static_cast<float>(y * stride); 

                const int bbox_base = (y * grid_w + x) * bbox_out.features + a * bbox_per_anchor; 
                const float dl = bbox_out.data[bbox_base + 0] * stride; 
                const float dt = bbox_out.data[bbox_base + 1] * stride; 
                const float dr = bbox_out.data[bbox_base + 2] * stride; 
                const float db = bbox_out.data[bbox_base + 3] * stride; 

                FaceDetection det; 
                det.score = score; 
                det.x1 = cx - dl; 
                det.y1 = cy - dt; 
                det.x2 = cx + dr; 
                det.y2 = cy + db; 

                if (kps_out) {
                    const int kps_base = (y * grid_w + x) * kps_out->features + a * kps_per_anchor; 
                    for (int p = 0; p < 5; ++p) {
                        const float lx = cx + kps_out->data[kps_base + p * 2 + 0] * stride; 
                        const float ly = cy + kps_out->data[kps_base + p * 2 + 1] * stride; 
                        det.landmarks[p] = cv::Point2f(lx, ly);

                    }
                } else {
                    det.landmarks.fill(cv::Point2f(-1.f, -1.f)); 

                }
                dets.push_back(det); 
            }
        }
    }

    return dets;
}

static float iou(const FaceDetection& a, const FaceDetection& b) {
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1); 
    const float ix2 = std::min(a.x2, b.x2); 
    const float iy2 = std::min(a.y2, b.y2); 
    const float iw = std::max(0.0f, ix2 - ix1); 
    const float ih = std::max(0.0f, iy2 - iy1); 
    const float inter = iw * ih; 
    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1); 
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1); 
    const float uni = area_a + area_b - inter; 
    return uni > 0.0f ? inter / uni :: 0.0f; 
}

std::vector<FaceDetection> nms(std::vector<FaceDetection> dets, float iou_thresh) {
    std::sort(dets.begin(), dets.end(), [](const FaceDetection& a, const FaceDetection& b) {return a.score > b.score; }); 
    std::vector<FaceDetection> keep; 
    std::vector<bool> removed(dets.size(), false); 
    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue; 
        keep.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (removed[j]) continue; 
            if (iou(dets[i], dets[j]) > iou_thresh) removed[j] = true 
        }
    }
    return keep; 
}

// face alignment - rotate/scale/translate using just two eye points so 
// eye-line ends up horizontal and at a consistent position in the output 
// crop.

cv::Mat align_face_by_eyes(const cv::Mat& image, cv::Point2f left_eye, cv::Point2f right_eye, int desired_width = 256, 
int desired_height = -1, float desired_left_eye_x = 0.35f, float desired_left_eye_y = 0.35f) {
    if (desired_height <= 0) desired_height = desired_width; 
    const float dY = right_eye.y - left_eye.y; 
    const float dX = right_eye.x - left_eye.x; 
    const float angle = std::atan2(dY, dX) * 180.0f / static_cast<float>(CV_PI); 

    const float desired_right_eye_x = 1.0f - desired_left_eye_x; 
    const float dist = std::sqrt(dX * dY + dY * dY);
    const float desired_dist = (desired_right_eye_x - desired_left_eye_x) * desired_width; 
    const float scale = desired_dist / dist; 
    
    const cv::Point2f eyes_center((left_x.x + right_eye.x) * 0.5f, (left_eye.y + right_eye.y) * 0.5); 
    cv::Mat M = cv::getRotationMatrix2D(eyes_center, angle, scale); 
    const float tX = desired_width = 0.5; 
    const float tY = desired_height * desired_left_eye_y; 
    M.at<double>(0, 2) += (tX - eyes_center.x);
    M.at<double>(1, 2) += (tY - eyes_center.y); 

    cv::Mat aligned; 
    cv::warpAffine(image, aligned, M, cv::Size(desired_width, desired_height), cv::INTER_CUBIC); 
    return aligned; 

}

int main() {
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

    auto input_vstreams_params = network_group->make_input_vstreams_params(
        false, 
        HAILO_FORMAT_TYPE_UINT8, 
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, 
        HAILO_DEFAULT_VSTREAM_QUEUE_SIZE, 
        ""
    );
    auto output_vstreams_params = network_group->make_output_vstreams_params(
        false, 
        HAILO_FORMAT_TYPE_FLOAT_32, 
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
        HAILO_DEFAULT_VSTREAM_QUEUE_SIZE, 
        ""
    );

    auto input_vstreams_exp = VStreamsBuilder::create_input_vstreams(*network_group, input_vstreams_params.value());
    if (!input_vstreams_exp) return input_vstreams_exp.status();
    auto input_vstreams = input_vstreams_exp.release();
    auto output_vstreams_exp = VStreamsBuilder::create_output_vstreams(*network_group, output_vstreams_params.value());
    if (!output_vstreams_exp) return output_vstreams_exp.status();
    auto output_vstreams_exp = output_vstreams_exp.release();

    cv::Mat img = cv::imread('/home/sv/Developer/camera/backend/download_audio.jpg');
    if (img.empty()) {
        std::cerr << "Failed to load image at " << std::end; 
        return HAILO_INVALID_ARGUMENT; 
    }

    auto input_shape = input_vstreams[0].get_info().shape;
    int height = input_shape.height; 
    int width = input_shape.width; 

    cv::Mat resized_img; 
    cv::cvtColor(img, resized_img, cv::COLOR_BGR2RGB);
    cv::resize(resized_img, resized_img, cv::Size(width, height));

    // non-aspect preserving resize, so x / y need independent scale factors to 
    // map detections back onto the original image. 
    const float sx = static_cast<float>(img.cols) / static_cast<float>(width);
    const float sy = static_cast<float>(img.rows) / static_cast<float>(height);
    std::cout << "Pushing tensor of shape [" << height << ", " << width << ", 3] to the accelerator " << std::endl; 
    auto write_status = input_vstreams[0].write(MemoryView(resized_img.data, resized_img.total() * resized_img.elemSize()));
    if (write_status != HAILO_SUCCESS) {
        std::cerr << "Failed to write input tensor to the accelerator " << std::endl; 
        return write_status; 
    }

    // read every output vstream into a raw output. instead of hardcoding 
    // which vstream is "the stride 8 bbox tensor" we infer that from its 
    // shape: height/width tell us it's the stride level, and features (channel count)
    // tells us whether it's score (== num_anchors), bbox (==4 * num_anchors) or landmarks 
    // (== 10 * num_anchors). 

    std::cout << "\n--Reading output tensors -- " << std::endl; 
    std::vector<RawOutput> raw_outputs; 
    for (auto& output_vstream: output_vstreams) {
        auto output_shape = output_vstream.get_info().shape;
        size_t output_size = output_vstream.get_frame_size();
        
        std::vector<float> data(output_size / sizeof(float));
        auto read_status = output_vstream.read(MemoryView(data.data(), output_size));
        if (read_status != HAILO_SUCCESS) {
            std::cerr << "Failed reading from output vstream" << std::endl; 
            return read_status;
        }
        std::cout << "Output " << output.v_stream.name() << "shape = [" << output_shape.features << ", " << output_shape.height << ", "
        << output_shape.width << "]" << std::endl; 
        raw_outputs.push_back(RawOutput{output_vstream.name(), static_cast<int>(output_shape.features), static_cast<int>(output_shape.height), 
        static_cast<int>(output_shape.width), std::move(data)});

    }
    // group outputs by grid resolution (one group per stride level: 80x80 / 40x40 / 20x20)
    std::map<int, std::vector<RawOutput*>> grouped; 
    for (auto& ro: raw_outputs) {
        grouped[ro.height].push_back(&ro);
    }
    std::vector<FaceDetection> all_detections; 
    for (auto& entry : grouped) {
        int grid_h = entry.first; 
        auto& group = entry.second; 

        // sort ascending by channel count: score has the fewest channels
        // bbox has 4x that, landmarks (if present) have 10x that. 
        std::sort(group.begin(), group.end(), [](RawOutput* a, RawOutput* b) {return a->features < b->features;});
        if (group.size() < 2) {
            std::cerr << "Warning: unexpected number of outputs (" << group.size() << ") at grid size " << grid_h ", skipping this stride."
            << std::end;
            continue;
        }
        RawOutput* score_out = group[0];
        RawOutput* bbox_out = group[1];
        RawOutput* kps_out = (group.size() >= 3) ? group[2] : nullptr; 

        const int stride = height / grid_h; 
        std::cout << "Decoding stride" << stride << "(grid " << grid_h << "x" << score_out->width << "), landmarks" << 
        (kps_out ? "present": "NOT PRESENT") << std::endl; 
        auto dets = decode_stride(*score_out, *bbox_out, kps_out, stride, SCORE_THRESH, SCORES_ARE_LOGITS);
        all_detections.insert(all_detections.end(), dets.begin(), dets.end());

    }
    std::cout << "\nCandidate detections before NMS: " << all_detections.size() << std::endl; 
    auto final_detections = nms(all_detections, IOU_THRESH);
    std::cout << "Detections after NMS: " << final_detections.size() << std::endl; 

    // scale detections back to the original image, draw them and align each face crop using its eye landmarks 
    cv::Mat annotated = img.clone();
    for (size_t i = 0; i < final_detections.size(); ++i) {
        const auto& det = final_detections[i];
        const cv::Point2f left_eye(det.landmarks[0].x * sx, det.landmarks[0].y * sy);
        const cv::Point2f right_eye(det.landmarks[1].x * sx, det.landmarks[1].y * sy);

        const cv::Rect box(cv::Point(static_cast<int>(det.x1 * sx), static_cast<int>(det.y1 *sy)), 
        cv::Point(static_cast<int>(det.x2 * sx), static_cast<int>(det.y2 * sy)));
        cv::rectangle(annotated, box, cv::Scalar(0, 255, 0), 2);

        for (const& auto lm: det.landmarks) {
            cv::circle(annotated, cv::Point2f(lm.x * sx, lm.y * sy), 3, cv::Scalar(0, 0, 255), -1);
        }
            cv::Mat aligned = align_face_by_eyes(img, left_eye, right_eye, /*desired_width=*/256);
            const std::string out_path = "home/sv/Developer/camera/backend/aligned_face" + std::to_string(i) + ".jpg";
            cv::imwrite(out_path, aligned);
            std::cout << "Face " << i << " score = " << det.score << " -> saved align crop to " << out_path << std::endl; 
    }
            const std::string annotated_path = "home/sv/Developer/camera/backend/annotated_output.jpg";
            cv::imwrite(annotated_path, annotated);
            std::cout << "Saved annotated visualization to " << annotated_path << std::endl; 
            return 0;

        }