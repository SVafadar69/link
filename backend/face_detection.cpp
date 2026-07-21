#include "hailo/hailort.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace hailort;


"""
score tensor: prob anchor has face
bbox tensor: left, top, right, bottom distances from the anchor 
landmark tensor: x/y offsets for facial 5 landmarks 

raw bbox outputs are distances from a known reference (anchor) point 
anchor_x = 240 
left bbox raw output = 4.0 -> this is NOT pixels. they are dimensionless multiples of the stride 
they are dimensionless multiples of the stride -> value of 1.0 = 1.0 x 8 (stride) = 8 pixels. 4.0 x 8 = 32 pixels away. 
stride = scale of pixel movements - stride of 8 = 8 pixels move in original feature map, 1 moves in stride 


raw bbox output of scrfd model = distances measured in units of the feature map stride 
to decode: anchor_x/y - position (top/bottom, left/right) * model feature map stride (one of 8, 16, 32).
80x80 -> stride 8 (smaller faces)
40x40 -> stride 16 (medium faces)
20x20 -> stride 20
"""

static configure_hailo_model() {}

struct Detection
{
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float score = 0.0f;

    // Landmark order:
    // 0 = left eye
    // 1 = right eye
    // 2 = nose
    // 3 = left mouth corner
    // 4 = right mouth corner
    std::array<cv::Point2f, 5> landmarks;
};


struct ScrfdScale
{
    std::string score_output;
    std::string bbox_output;
    std::string landmark_output;

    int height;
    int width;
    int stride;
};


static bool ends_with(
    const std::string &value,
    const std::string &suffix)
{
    if (suffix.size() > value.size()) {
        return false;
    }

    return std::equal(
        suffix.rbegin(),
        suffix.rend(),
        value.rbegin()
    );
}

// iou between 2 detection s

static float intersection_over_union(
    const Detection &a,
    const Detection &b)
{
    const float intersection_left =
        std::max(a.x1, b.x1);

    const float intersection_top =
        std::max(a.y1, b.y1);

    const float intersection_right =
        std::min(a.x2, b.x2);

    const float intersection_bottom =
        std::min(a.y2, b.y2);

    const float intersection_width =
        std::max(
            0.0f,
            intersection_right - intersection_left
        );

    const float intersection_height =
        std::max(
            0.0f,
            intersection_bottom - intersection_top
        );

    const float intersection_area =
        intersection_width * intersection_height;

    const float area_a =
        std::max(0.0f, a.x2 - a.x1) *
        std::max(0.0f, a.y2 - a.y1);

    const float area_b =
        std::max(0.0f, b.x2 - b.x1) *
        std::max(0.0f, b.y2 - b.y1);

    const float union_area =
        area_a + area_b - intersection_area;

    if (union_area <= 0.0f) {
        return 0.0f;
    }

    return intersection_area / union_area;
}

static std::vector<Detection> non_maximum_suppression(
    const std::vector<Detection> &detections,
    float iou_threshold)
{
    std::vector<size_t> indices(detections.size());

    std::iota(
        indices.begin(),
        indices.end(),
        0
    );

    std::sort(
        indices.begin(),
        indices.end(),
        [&](size_t a, size_t b) {
            return detections[a].score >
                   detections[b].score;
        }
    );

    std::vector<Detection> kept_detections;

    for (const size_t candidate_index : indices) {
        const Detection &candidate =
            detections[candidate_index];

        bool should_keep = true;

        for (const Detection &kept : kept_detections) {
            if (intersection_over_union(
                    candidate,
                    kept) > iou_threshold) {

                should_keep = false;
                break;
            }
        }

        if (should_keep) {
            kept_detections.push_back(candidate);
        }
    }

    return kept_detections;
}

// decoding scrfd scale
// what is is 
// the math behind it 
// information about the actual detector (get in advance)

static void decode_scrfd_scale(
    const float *score_data,
    const float *bbox_data,
    const float *landmark_data,
    int feature_height,
    int feature_width,
    int stride,
    int model_width,
    int model_height,
    int original_width,
    int original_height,
    float score_threshold,
    std::vector<Detection> &detections)
{

    // scrfd_10g has two prediction anchors per grid cell.
    constexpr int ANCHORS_PER_CELL = 2;

    const int score_channels =
        ANCHORS_PER_CELL;

    const int bbox_channels =
        ANCHORS_PER_CELL *
        BBOX_VALUES_PER_ANCHOR;

    const int landmark_channels =
        ANCHORS_PER_CELL *
        LANDMARK_VALUES_PER_ANCHOR;

    const float original_scale_x =
        static_cast<float>(original_width) /
        static_cast<float>(model_width);

    const float original_scale_y =
        static_cast<float>(original_height) /
        static_cast<float>(model_height);

    for (int y = 0; y < feature_height; ++y) {
        for (int x = 0; x < feature_width; ++x) {
            const int cell_index =
                y * feature_width + x;

            /*
             * Hailo's official SCRFD postprocessor places the
             * anchor centre at:
             *
             *     center_x = x * stride
             *     center_y = y * stride
             *
             * It does not add 0.5 to x or y.
             */
            const float anchor_center_x =
                static_cast<float>(x * stride);

            const float anchor_center_y =
                static_cast<float>(y * stride);

            for (int anchor = 0;
                 anchor < ANCHORS_PER_CELL;
                 ++anchor) {

                const int score_offset =
                    cell_index * score_channels +
                    anchor;

                /*
                 * The HEF's classification output is already
                 * the face score. Do not apply sigmoid again.
                 */
                const float score =
                    score_data[score_offset];

                if (score < score_threshold) {
                    continue;
                }

                const int bbox_offset =
                    cell_index * bbox_channels +
                    anchor * BBOX_VALUES_PER_ANCHOR;

                // Raw bbox distances from the anchor centre.
                const float left_distance =
                    bbox_data[bbox_offset];

                const float top_distance =
                    bbox_data[bbox_offset + 1];

                const float right_distance =
                    bbox_data[bbox_offset + 2];

                const float bottom_distance =
                    bbox_data[bbox_offset + 3];

                float x1_model =
                    anchor_center_x -
                    left_distance *
                        static_cast<float>(stride);

                float y1_model =
                    anchor_center_y -
                    top_distance *
                        static_cast<float>(stride);

                float x2_model =
                    anchor_center_x +
                    right_distance *
                        static_cast<float>(stride);

                float y2_model =
                    anchor_center_y +
                    bottom_distance *
                        static_cast<float>(stride);

                x1_model = std::clamp(
                    x1_model,
                    0.0f,
                    static_cast<float>(model_width - 1)
                );

                y1_model = std::clamp(
                    y1_model,
                    0.0f,
                    static_cast<float>(model_height - 1)
                );

                x2_model = std::clamp(
                    x2_model,
                    0.0f,
                    static_cast<float>(model_width - 1)
                );

                y2_model = std::clamp(
                    y2_model,
                    0.0f,
                    static_cast<float>(model_height - 1)
                );

                if (x2_model <= x1_model ||
                    y2_model <= y1_model) {

                    continue;
                }

                Detection detection;

                // Map the model coordinates back to the original image.
                detection.x1 =
                    x1_model * original_scale_x;

                detection.y1 =
                    y1_model * original_scale_y;

                detection.x2 =
                    x2_model * original_scale_x;

                detection.y2 =
                    y2_model * original_scale_y;

                detection.score = score;

                const int landmark_offset =
                    cell_index * landmark_channels +
                    anchor * LANDMARK_VALUES_PER_ANCHOR;

                for (int landmark_index = 0;
                     landmark_index < 5;
                     ++landmark_index) {

                    const float x_offset =
                        landmark_data[
                            landmark_offset +
                            landmark_index * 2
                        ];

                    const float y_offset =
                        landmark_data[
                            landmark_offset +
                            landmark_index * 2 + 1
                        ];

                    /*
                     * Landmark decoding:
                     *
                     * landmark_x = anchor_x + raw_x * stride
                     * landmark_y = anchor_y + raw_y * stride
                     */
                    float landmark_x_model =
                        anchor_center_x +
                        x_offset *
                            static_cast<float>(stride);

                    float landmark_y_model =
                        anchor_center_y +
                        y_offset *
                            static_cast<float>(stride);

                    landmark_x_model = std::clamp(
                        landmark_x_model,
                        0.0f,
                        static_cast<float>(model_width - 1)
                    );

                    landmark_y_model = std::clamp(
                        landmark_y_model,
                        0.0f,
                        static_cast<float>(model_height - 1)
                    );

                    detection.landmarks[landmark_index] =
                        cv::Point2f(
                            landmark_x_model *
                                original_scale_x,

                            landmark_y_model *
                                original_scale_y
                        );
                }

                detections.push_back(detection);
            }
        }
    }
}

// what is std::clamp - why is clamp being used here

static void draw_detection(
    cv::Mat &image,
    const Detection &detection,
    size_t detection_index)
{
    const int x1 = std::clamp(
        static_cast<int>(std::round(detection.x1)),
        0,
        image.cols - 1
    );

    const int y1 = std::clamp(
        static_cast<int>(std::round(detection.y1)),
        0,
        image.rows - 1
    );

    const int x2 = std::clamp(
        static_cast<int>(std::round(detection.x2)),
        0,
        image.cols - 1
    );

    const int y2 = std::clamp(
        static_cast<int>(std::round(detection.y2)),
        0,
        image.rows - 1
    );

    cv::rectangle(
        image,
        cv::Point(x1, y1),
        cv::Point(x2, y2),
        cv::Scalar(0, 255, 0),
        2,
        cv::LINE_AA
    );

    std::ostringstream label; 
    label 
        << "face "
        << std::fixed 
        << std::setprecision(2)
        << detection.score; 


    int baseline = 0;

    const cv::Size text_size =
        cv::getTextSize(
            label.str(),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
            &baseline
        );

    const int label_x =
        std::clamp(
            x1,
            0,
            std::max(0, image.cols - text_size.width - 8)
        );

    const int label_y =
        std::max(
            0,
            y1 - text_size.height - 10
        );

    cv::rectangle(
        image,
        cv::Rect(
            label_x,
            label_y,
            text_size.width + 8,
            text_size.height + 10
        ),
        cv::Scalar(0, 255, 0),
        cv::FILLED
    );

    cv::putText(
        image,
        label.str(),
        cv::Point(
            label_x + 4,
            label_y + text_size.height + 3
        ),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(0, 0, 0),
        2,
        cv::LINE_AA
    );

    // Left eye.
    cv::circle(
        image,
        detection.landmarks[0],
        4,
        cv::Scalar(0, 0, 255),
        cv::FILLED,
        cv::LINE_AA
    );

    // Right eye.
    cv::circle(
        image,
        detection.landmarks[1],
        4,
        cv::Scalar(255, 0, 0),
        cv::FILLED,
        cv::LINE_AA
    );

    // Nose and mouth corners.
    for (int landmark_index = 2;
         landmark_index < 5;
         ++landmark_index) {

        cv::circle(
            image,
            detection.landmarks[landmark_index],
            3,
            cv::Scalar(0, 255, 255),
            cv::FILLED,
            cv::LINE_AA
        );
    }

    // Draw the eye line.
    cv::line(
        image,
        detection.landmarks[0],
        detection.landmarks[1],
        cv::Scalar(255, 255, 0),
        2,
        cv::LINE_AA
    );

    const cv::Point2f left_eye =
        detection.landmarks[0];

    const cv::Point2f right_eye =
        detection.landmarks[1];

    const float eye_angle_degrees =
        static_cast<float>(
            std::atan2(
                right_eye.y - left_eye.y,
                right_eye.x - left_eye.x
            ) *
            180.0 /
            CV_PI
        );

    std::cout
        << "Face " << detection_index
        << "\n  score: "
        << detection.score
        << "\n  bbox: ["
        << detection.x1 << ", "
        << detection.y1 << ", "
        << detection.x2 << ", "
        << detection.y2 << "]"
        << "\n  left eye: ("
        << left_eye.x << ", "
        << left_eye.y << ")"
        << "\n  right eye: ("
        << right_eye.x << ", "
        << right_eye.y << ")"
        << "\n  nose: ("
        << detection.landmarks[2].x << ", "
        << detection.landmarks[2].y << ")"
        << "\n  left mouth: ("
        << detection.landmarks[3].x << ", "
        << detection.landmarks[3].y << ")"
        << "\n  right mouth: ("
        << detection.landmarks[4].x << ", "
        << detection.landmarks[4].y << ")"
        << "\n  eye-line angle: "
        << eye_angle_degrees
        << " degrees\n";
}

int main(int argc, char **argv)
{

    const std::string image_path =
        argv[1];

    const std::string output_path =
        argc >= 3
            ? argv[2]
            : "scrfd_result.jpg";

     if (argc < 2) {
        std::cerr
            << "Usage: "
            << argv[0]
            << " download_audio.jpg [output.jpg]\n";

        return 1;
    }

    constexpr float SCORE_THRESHOLD = 0.50f;
    constexpr float NMS_IOU_THRESHOLD = 0.40f;


    cv::Mat original_image =
        cv::imread(image_path);

    cv::Mat annotated_image =
        original_image.clone();
    

    const std::string hef_path =
        "/home/sv/Developer/camera/backend/"
        "hailo8l_models/scrfd_10g.hef";

    auto vdevice_exp =
        VDevice::create();

    if (!vdevice_exp) {
        std::cerr
            << "Failed to create VDevice. Status: "
            << vdevice_exp.status()
            << '\n';

        return 1;
    }

    auto vdevice =
        vdevice_exp.release();

    // ---------------------------------------------------------
    // Create inference model
    // ---------------------------------------------------------

    auto infer_model_exp =
        vdevice->create_infer_model(hef_path);

    if (!infer_model_exp) {
        std::cerr
            << "Failed to create InferModel. Status: "
            << infer_model_exp.status()
            << '\n';

        return 1;
    }

    auto infer_model =
        infer_model_exp.release();

    // ---------------------------------------------------------
    // Request dequantized FLOAT32 outputs
    // ---------------------------------------------------------

    for (const std::string &output_name :
         infer_model->get_output_names()) {

        auto output_exp =
            infer_model->output(output_name);

        if (!output_exp) {
            std::cerr
                << "Failed to access output: "
                << output_name
                << '\n';

            return 1;
        }

        /*
         * In your installed HailoRT version,
         * set_format_type() returns void.
         */
        output_exp->set_format_type(
            HAILO_FORMAT_TYPE_FLOAT32
        );
    }

    // ---------------------------------------------------------
    // Configure model
    // ---------------------------------------------------------

    auto configured_model_exp =
        infer_model->configure();

    if (!configured_model_exp) {
        std::cerr
            << "Failed to configure model. Status: "
            << configured_model_exp.status()
            << '\n';

        return 1;
    }

    auto configured_model =
        configured_model_exp.release();

    auto bindings_exp =
        configured_model.create_bindings();

    if (!bindings_exp) {
        std::cerr
            << "Failed to create bindings. Status: "
            << bindings_exp.status()
            << '\n';

        return 1;
    }

    auto bindings =
        bindings_exp.release();

    // ---------------------------------------------------------
    // Read input information
    // ---------------------------------------------------------

    auto model_inputs =
        infer_model->inputs();

    if (model_inputs.size() != 1) {
        std::cerr
            << "Expected one model input, found "
            << model_inputs.size()
            << '\n';

        return 1;
    }

    const auto input_shape =
        model_inputs[0].shape();

    const int model_height =
        static_cast<int>(input_shape.height);

    const int model_width =
        static_cast<int>(input_shape.width);

    const int model_channels =
        static_cast<int>(input_shape.features);

    std::cout
        << "Input name: "
        << model_inputs[0].name()
        << '\n'
        << "Input shape: ["
        << model_height << ", "
        << model_width << ", "
        << model_channels << "]\n";

    if (model_channels != 3) {
        std::cerr
            << "Expected three input channels, found "
            << model_channels
            << '\n';

        return 1;
    }

    // ---------------------------------------------------------
    // Preprocess image
    // ---------------------------------------------------------

    cv::Mat resized_image;

    cv::resize(
        original_image,
        resized_image,
        cv::Size(model_width, model_height)
    );

    cv::Mat rgb_image;

    cv::cvtColor(
        resized_image,
        rgb_image,
        cv::COLOR_BGR2RGB
    );

    if (!rgb_image.isContinuous()) {
        rgb_image = rgb_image.clone();
    }

    const size_t expected_input_bytes =
        model_inputs[0].get_frame_size();

    const size_t actual_input_bytes =
        rgb_image.total() *
        rgb_image.elemSize();

    if (actual_input_bytes != expected_input_bytes) {
        std::cerr
            << "Input byte-size mismatch.\n"
            << "Model expects: "
            << expected_input_bytes
            << " bytes\n"
            << "Prepared image has: "
            << actual_input_bytes
            << " bytes\n";

        return 1;
    }

    std::vector<uint8_t> input_buffer(
        expected_input_bytes
    );

    std::memcpy(
        input_buffer.data(),
        rgb_image.data,
        expected_input_bytes
    );

    hailo_status input_status =
        bindings
            .input(model_inputs[0].name())
            ->set_buffer(
                MemoryView(
                    input_buffer.data(),
                    input_buffer.size()
                )
            );

    if (input_status != HAILO_SUCCESS) {
        std::cerr
            << "Failed to bind input buffer. Status: "
            << input_status
            << '\n';

        return 1;
    }

    // ---------------------------------------------------------
    // Allocate output buffers
    // ---------------------------------------------------------

    auto model_outputs =
        infer_model->outputs();

    std::vector<std::vector<float>>
        output_buffers(model_outputs.size());

    for (size_t i = 0;
         i < model_outputs.size();
         ++i) {

        const size_t output_bytes =
            model_outputs[i].get_frame_size();

        if (output_bytes % sizeof(float) != 0) {
            std::cerr
                << "Output is not FLOAT32-compatible: "
                << model_outputs[i].name()
                << '\n';

            return 1;
        }

        output_buffers[i].resize(
            output_bytes / sizeof(float)
        );

        hailo_status output_status =
            bindings
                .output(model_outputs[i].name())
                ->set_buffer(
                    MemoryView(
                        output_buffers[i].data(),
                        output_bytes
                    )
                );

        if (output_status != HAILO_SUCCESS) {
            std::cerr
                << "Failed to bind output: "
                << model_outputs[i].name()
                << ". Status: "
                << output_status
                << '\n';

            return 1;
        }
    }

    // ---------------------------------------------------------
    // Run inference
    // ---------------------------------------------------------

    const hailo_status inference_status =
        configured_model.run(
            bindings,
            std::chrono::milliseconds(1000)
        );

    if (inference_status != HAILO_SUCCESS) {
        std::cerr
            << "Inference failed. Status: "
            << inference_status
            << '\n';

        return 1;
    }

    std::cout
        << "Inference completed successfully.\n";

    // ---------------------------------------------------------
    // Print outputs and locate tensors
    // ---------------------------------------------------------

    for (size_t i = 0;
         i < model_outputs.size();
         ++i) {

        const auto shape =
            model_outputs[i].shape();

        std::cout
            << "Output: "
            << model_outputs[i].name()
            << "\nShape: ["
            << shape.height << ", "
            << shape.width << ", "
            << shape.features << "]"
            << "\nElements: "
            << output_buffers[i].size()
            << "\n";
    }

    auto find_output =
        [&](const std::string &suffix,
            int expected_height,
            int expected_width,
            int expected_channels) -> const float * {

        for (size_t i = 0;
             i < model_outputs.size();
             ++i) {

            const auto shape =
                model_outputs[i].shape();

            if (!ends_with(
                    model_outputs[i].name(),
                    suffix)) {

                continue;
            }

            if (static_cast<int>(shape.height) !=
                    expected_height ||
                static_cast<int>(shape.width) !=
                    expected_width ||
                static_cast<int>(shape.features) !=
                    expected_channels) {

                std::cerr
                    << "Unexpected shape for "
                    << model_outputs[i].name()
                    << ". Expected ["
                    << expected_height << ", "
                    << expected_width << ", "
                    << expected_channels << "], got ["
                    << shape.height << ", "
                    << shape.width << ", "
                    << shape.features << "]\n";

                return nullptr;
            }

            return output_buffers[i].data();
        }

        std::cerr
            << "Could not find output ending with: "
            << suffix
            << '\n';

        return nullptr;
    };

    const std::vector<ScrfdScale> scales = {
        {
            "/conv41",
            "/conv42",
            "/conv43",
            80,
            80,
            8
        },
        {
            "/conv49",
            "/conv50",
            "/conv51",
            40,
            40,
            16
        },
        {
            "/conv56",
            "/conv57",
            "/conv58",
            20,
            20,
            32
        }
    };

    std::vector<Detection> candidate_detections;

    candidate_detections.reserve(16800);

    // ---------------------------------------------------------
    // Decode all three scales
    // ---------------------------------------------------------

    for (const ScrfdScale &scale : scales) {
        const float *scores =
            find_output(
                scale.score_output,
                scale.height,
                scale.width,
                2
            );

        const float *boxes =
            find_output(
                scale.bbox_output,
                scale.height,
                scale.width,
                8
            );

        const float *landmarks =
            find_output(
                scale.landmark_output,
                scale.height,
                scale.width,
                20
            );

        if (scores == nullptr ||
            boxes == nullptr ||
            landmarks == nullptr) {

            std::cerr
                << "Missing one or more tensors for stride "
                << scale.stride
                << '\n';

            return 1;
        }

        decode_scrfd_scale(
            scores,
            boxes,
            landmarks,
            scale.height,
            scale.width,
            scale.stride,
            model_width,
            model_height,
            original_image.cols,
            original_image.rows,
            SCORE_THRESHOLD,
            candidate_detections
        );
    }

    std::cout
        << "Candidates above threshold: "
        << candidate_detections.size()
        << '\n';

    // ---------------------------------------------------------
    // Non-maximum suppression
    // ---------------------------------------------------------

    const std::vector<Detection> final_detections =
        non_maximum_suppression(
            candidate_detections,
            NMS_IOU_THRESHOLD
        );

    std::cout
        << "Faces after NMS: "
        << final_detections.size()
        << '\n';

    // ---------------------------------------------------------
    // Draw detections
    // ---------------------------------------------------------

    for (size_t i = 0;
         i < final_detections.size();
         ++i) {

        draw_detection(
            annotated_image,
            final_detections[i],
            i
        );
    }

    // ---------------------------------------------------------
    // Save and display result
    // ---------------------------------------------------------

    if (!cv::imwrite(
            output_path,
            annotated_image)) {

        std::cerr
            << "Failed to save output image: "
            << output_path
            << '\n';

        return 1;
    }

    std::cout
        << "Saved annotated image to: "
        << output_path
        << '\n';

    cv::imshow(
        "SCRFD detections",
        annotated_image
    );

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}