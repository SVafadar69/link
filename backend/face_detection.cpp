#include "hailo/hailort.hpp"
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace hailort;

//arcface_r50.hef -> 112x112x3 input image as input; rgb; uint8 0-255 

const std::string face_embeddings_model = "arcface_r50.hef";

constexpr float SCORE_THRESHOLD = 0.50f;
constexpr float NMS_THRESHOLD = 0.40f;

struct Detection {
    float x1, y1, x2, y2, score;
    std::array<cv::Point2f, 5> landmarks;
};

struct Scale {
    const char *scores;
    const char *boxes;
    const char *landmarks;
    int height;
    int width;
    int stride;
};

struct HailoModel {
    std::unique_ptr<VDevice> vdevice;
    std::shared_ptr<InferModel> infer_model;
    ConfiguredInferModel configured_model;
};

static HailoModel load_hailo_model(const std::string &hef_path)
{
    auto vdevice_exp = VDevice::create();

    if (!vdevice_exp) {
        throw std::runtime_error("Failed to create VDevice");
    }

    auto vdevice = vdevice_exp.release();

    auto infer_model_exp =
        vdevice->create_infer_model(hef_path);

    if (!infer_model_exp) {
        throw std::runtime_error("Failed to load HEF");
    }

    auto infer_model = infer_model_exp.release();

    for (const auto &name : infer_model->get_output_names()) {
        auto output = infer_model->output(name);

        if (!output) {
            throw std::runtime_error(
                "Failed to access output: " + name
            );
        }

        output->set_format_type(
            HAILO_FORMAT_TYPE_FLOAT32
        );
    }

    auto configured_exp = infer_model->configure();

    if (!configured_exp) {
        throw std::runtime_error(
            "Failed to configure model"
        );
    }

    return {
        std::move(vdevice),
        std::move(infer_model),
        configured_exp.release()
    };
}

static void l2_normalize(std::vector<float> &embedding) {
    float squared_sum = 0.0f;
    for (float_value: embedding) {
        squared_sum += value * value; 
    }

    const float norm = std::sqrt(squared_sum);

    if (norm > 0.0f) {
        for (float &value : embedding) {
            value /= norm;
        }
    }
}

static float iou(const Detection &a, const Detection &b)
{
    const float left = std::max(a.x1, b.x1);
    const float top = std::max(a.y1, b.y1);
    const float right = std::min(a.x2, b.x2);
    const float bottom = std::min(a.y2, b.y2);

    const float intersection =
        std::max(0.0f, right - left) *
        std::max(0.0f, bottom - top);

    const float area_a =
        std::max(0.0f, a.x2 - a.x1) *
        std::max(0.0f, a.y2 - a.y1);

    const float area_b =
        std::max(0.0f, b.x2 - b.x1) *
        std::max(0.0f, b.y2 - b.y1);

    const float union_area =
        area_a + area_b - intersection;

    return union_area > 0.0f
        ? intersection / union_area
        : 0.0f;
}

static std::vector<Detection> nms(
    const std::vector<Detection> &detections)
{
    std::vector<size_t> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(
        indices.begin(),
        indices.end(),
        [&](size_t a, size_t b) {
            return detections[a].score >
                   detections[b].score;
        }
    );

    std::vector<Detection> kept;

    for (size_t index : indices) {
        bool overlaps = false;

        for (const auto &existing : kept) {
            if (iou(detections[index], existing) >
                NMS_THRESHOLD) {

                overlaps = true;
                break;
            }
        }

        if (!overlaps) {
            kept.push_back(detections[index]);
        }
    }

    return kept;
}

static void decode_scale(
    const float *scores,
    const float *boxes,
    const float *landmarks,
    int feature_height,
    int feature_width,
    int stride,
    int model_width,
    int model_height,
    int image_width,
    int image_height,
    std::vector<Detection> &detections)
{
    constexpr int ANCHORS = 2;

    const float scale_x =
        static_cast<float>(image_width) / model_width;

    const float scale_y =
        static_cast<float>(image_height) / model_height;

    for (int y = 0; y < feature_height; ++y) {
        for (int x = 0; x < feature_width; ++x) {
            const int cell = y * feature_width + x;

            const float anchor_x =
                static_cast<float>(x * stride);

            const float anchor_y =
                static_cast<float>(y * stride);

            for (int anchor = 0; anchor < ANCHORS; ++anchor) {
                const float score =
                    scores[cell * 2 + anchor];

                if (score < SCORE_THRESHOLD) {
                    continue;
                }

                const int box_offset =
                    cell * 8 + anchor * 4;

                float x1 =
                    anchor_x - boxes[box_offset] * stride;

                float y1 =
                    anchor_y - boxes[box_offset + 1] * stride;

                float x2 =
                    anchor_x + boxes[box_offset + 2] * stride;

                float y2 =
                    anchor_y + boxes[box_offset + 3] * stride;

                x1 = std::clamp(x1, 0.0f, float(model_width - 1));
                y1 = std::clamp(y1, 0.0f, float(model_height - 1));
                x2 = std::clamp(x2, 0.0f, float(model_width - 1));
                y2 = std::clamp(y2, 0.0f, float(model_height - 1));

                if (x2 <= x1 || y2 <= y1) {
                    continue;
                }

                Detection detection {
                    x1 * scale_x,
                    y1 * scale_y,
                    x2 * scale_x,
                    y2 * scale_y,
                    score,
                    {}
                };

                const int landmark_offset =
                    cell * 20 + anchor * 10;

                for (int i = 0; i < 5; ++i) {
                    float landmark_x =
                        (
                            anchor_x +
                            landmarks[
                                landmark_offset + i * 2
                            ] * stride
                        ) * scale_x;

                    float landmark_y =
                        (
                            anchor_y +
                            landmarks[
                                landmark_offset + i * 2 + 1
                            ] * stride
                        ) * scale_y;

                    detection.landmarks[i] = {
                        std::clamp(
                            landmark_x,
                            0.0f,
                            float(image_width - 1)
                        ),
                        std::clamp(
                            landmark_y,
                            0.0f,
                            float(image_height - 1)
                        )
                    };
                }

                detections.push_back(detection);
            }
        }
    }
}

static cv::Mat face_alignment(
    const cv::Mat &frame, 
    const Detection &face, 
    float &angle_degrees
) {
    const int x1 = std::clamp(
        static_cast<int>(std::floor(face.x1)), 
        0, 
        frame.cols - 1
    );

    const int y1 = std::clamp(
        static_cast<int>(std::floor(face.y1)),
        0,
        frame.rows - 1
    );

    const int x2 = std::clamp(
        static_cast<int>(std::ceil(face.x2)),
        1,
        frame.cols
    );

    const int y2 = std::clamp(
        static_cast<int>(std::ceil(face.y2)),
        1,
        frame.rows
    );

    if (x2 <= x1 || y2 <= y1) {
        return {};
    }

    const cv::Rect face_roi(
        x1, 
        y1, 
        x2 - x1, 
        y2 - y1
    );

    cv::Mat face_crop = frame(face_roi).clone();

    cv::Point2f eye_1 = face.landmarks[0] - cv::Point2f(static_cast<float>(x1), static_cast<float>(y1));
    cv::Point2f eye_2 = face.landmarks[1] - cv::Point2f(static_cast<float>(x1), static_cast<float>(y1));

    // sort by image position instead of relying on labels 
    const cv::Point2f image_left_eye = eye_1.x < eye_2.x ? eye_1 : eye_2; 
    const cv::Point2f image_right_eye = eye_1.x < eye_2.x ? eye_2 : eye_1; 
    const float delta_x = image_right_eye.x - image_left_eye.x; 
    const float delta_y = image_right_eye.y - image_left_eye.y; 

    angle_degrees = std::atan2(delta_y, delta_x) * 180.0f / static_cast<float>(CV_PI);
    const cv::Point2f eye_center((image_left_eye.x + image_right_eye.x) * 0.5f, (image_left_eye.y + image_right_eye.y) * 0.5f);
    const cv::Mat rotation_matrix = cv::getRotationMatrix2D(eye_center, angle_degrees, 1.0);
    cv::Mat aligned_face; 
    cv::warpAffine(face_crop, aligned_face, rotation_matrix, face_crop.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    
    return aligned_face;

}

static void draw_detections(
    cv::Mat &frame,
    const std::vector<Detection> &detections)
{
    for (const auto &detection : detections) {
        cv::rectangle(
            frame,
            cv::Point(
                static_cast<int>(detection.x1),
                static_cast<int>(detection.y1)
            ),
            cv::Point(
                static_cast<int>(detection.x2),
                static_cast<int>(detection.y2)
            ),
            cv::Scalar(0, 255, 0),
            2
        );

        for (const auto &point : detection.landmarks) {
            cv::circle(
                frame,
                point,
                3,
                cv::Scalar(0, 0, 255),
                cv::FILLED
            );
        }

        const cv::Point2f &left_eye =
            detection.landmarks[0];

            std::cout << "Left eye landmark" << left_eye << std::endl;

        const cv::Point2f &right_eye =
            detection.landmarks[1];

            std::cout << "Right eye landmark" << right_eye << std::endl;

        const float angle =
            std::atan2(
                right_eye.y - left_eye.y,
                right_eye.x - left_eye.x
            ) *
            180.0f /
            static_cast<float>(CV_PI);

        cv::putText(
            frame,
            cv::format(
                "%.2f | %.1f deg",
                detection.score,
                angle
            ),
            cv::Point(
                static_cast<int>(detection.x1),
                std::max(
                    20,
                    static_cast<int>(detection.y1) - 5
                )
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 255, 0),
            2
        );
    }
}

static int run_camera(int camera_index)
{
    const std::string hef_path =
        "/home/sv/Developer/camera/backend/"
        "hailo8l_models/scrfd_10g.hef";

    HailoModel model =
        load_hailo_model(hef_path);

    auto bindings_exp =
        model.configured_model.create_bindings();

    if (!bindings_exp) {
        std::cerr << "Failed to create bindings\n";
        return 1;
    }

    auto bindings = bindings_exp.release();

    auto inputs = model.infer_model->inputs();
    auto outputs = model.infer_model->outputs();

    const auto input_shape = inputs[0].shape();

    const int model_height =
        static_cast<int>(input_shape.height);

    const int model_width =
        static_cast<int>(input_shape.width);

    std::vector<uint8_t> input_buffer(
        inputs[0].get_frame_size()
    );

    bindings.input(inputs[0].name())->set_buffer(
        MemoryView(
            input_buffer.data(),
            input_buffer.size()
        )
    );

    std::vector<std::vector<float>>
        output_buffers(outputs.size());

    for (size_t i = 0; i < outputs.size(); ++i) {
        const size_t bytes =
            outputs[i].get_frame_size();

        output_buffers[i].resize(
            bytes / sizeof(float)
        );

        bindings.output(outputs[i].name())->set_buffer(
            MemoryView(
                output_buffers[i].data(),
                bytes
            )
        );
    }

    auto get_output =
        [&](const std::string &suffix) -> const float * {

        for (size_t i = 0; i < outputs.size(); ++i) {
            const std::string name =
                outputs[i].name();

            if (name.size() >= suffix.size() &&
                name.compare(
                    name.size() - suffix.size(),
                    suffix.size(),
                    suffix
                ) == 0) {

                return output_buffers[i].data();
            }
        }

        return nullptr;
    };

    const std::array<Scale, 3> scales {{
        {"/conv41", "/conv42", "/conv43", 80, 80, 8},
        {"/conv49", "/conv50", "/conv51", 40, 40, 16},
        {"/conv56", "/conv57", "/conv58", 20, 20, 32}
    }};

    cv::VideoCapture camera(camera_index);

    if (!camera.isOpened()) {
        std::cerr
            << "Failed to open camera "
            << camera_index << '\n';

        return 1;
    }

    cv::Mat frame;
    cv::Mat resized;
    cv::Mat rgb;

    while (camera.read(frame)) {
        cv::resize(
            frame,
            resized,
            cv::Size(model_width, model_height)
        );

        cv::cvtColor(
            resized,
            rgb,
            cv::COLOR_BGR2RGB
        );

        if (!rgb.isContinuous()) {
            rgb = rgb.clone();
        }

        std::memcpy(
            input_buffer.data(),
            rgb.data,
            input_buffer.size()
        );

        const auto start =
            std::chrono::steady_clock::now();

        if (model.configured_model.run(
                bindings,
                std::chrono::milliseconds(1000)
            ) != HAILO_SUCCESS) {

            std::cerr << "Inference failed\n";
            break;
        }

        std::vector<Detection> detections;

        for (const auto &scale : scales) {
            const float *scores =
                get_output(scale.scores);

            const float *boxes =
                get_output(scale.boxes);

            const float *landmarks =
                get_output(scale.landmarks);

            if (!scores || !boxes || !landmarks) {
                std::cerr
                    << "Missing SCRFD output tensor\n";

                return 1;
            }

            decode_scale(
                scores,
                boxes,
                landmarks,
                scale.height,
                scale.width,
                scale.stride,
                model_width,
                model_height,
                frame.cols,
                frame.rows,
                detections
            );
        }

        detections = nms(detections);
        if (!detections.empty()) {
            float alignment_angle = 0.0f;
            cv::Mat aligned_face = face_alignment(frame, detections[0], alignment_angle);
            std::cout << "aligned face" << aligned_face.size() << std::endl;
            // converting to RGB for arcface_50 inference 
            cv::cvtColor(aligned_face, rgb_face, cv::COLOR_BGR2RGB);
            if (!rgb_face.isContinuous()) {
                rgb_face = rgb_face.clone();
                std::memcpy(arcface_input_buffer.data(), rgb_face.data(), arcface_input_buffer.size());
            }
            if (!aligned_face.empty()) {
                cv::putText(aligned_face, cv::format("Rotation: %.1f deg", alignment_angle),
                cv::Point2f(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                cv::imshow("Aligned Face", aligned_face);
            }
        }
        draw_detections(frame, detections);

        const auto end =
            std::chrono::steady_clock::now();

        const double milliseconds =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        const double fps =
            milliseconds > 0.0
                ? 1000.0 / milliseconds
                : 0.0;

        cv::putText(
            frame,
            cv::format(
                "Faces: %d | %.1f FPS",
                static_cast<int>(detections.size()),
                fps
            ),
            cv::Point(15, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(0, 255, 255),
            2
        );

        //cv::imshow("SCRFD Camera", frame);

        const int key = cv::waitKey(1);

        if (key == 27 || key == 'q') {
            break;
        }
    }

    camera.release();
    cv::destroyAllWindows();

    return 0;
}

int main(int argc, char **argv)
{
    try {
        const int camera_index =
            argc >= 2 ? std::stoi(argv[1]) : 0;

        return run_camera(camera_index);
    }
    catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}