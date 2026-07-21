#include "hailo/hailort.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace hailort;

// explain static - var + function 
// math behind decoding scale - get official source + ChatGPT math

struct ScaleOutput {
    int height;
    int width;

    std::vector<float> min_sizes;

    const float *bbox_data;
    const float *class_data;
};

static float face_softmax(float background_logit, float face_logit)
{
    // Numerically stable softmax for:
    // [background, face]
    const float maximum =
        std::max(background_logit, face_logit);

    const float background_exp =
        std::exp(background_logit - maximum);

    const float face_exp =
        std::exp(face_logit - maximum);

    return face_exp / (background_exp + face_exp);
}

static void decode_scale(
    const ScaleOutput &scale,
    int model_width,
    int model_height,
    int original_width,
    int original_height,
    float score_threshold,
    std::vector<cv::Rect> &boxes,
    std::vector<float> &scores)
{
    const int anchors_per_cell =
        static_cast<int>(scale.min_sizes.size());

    const int bbox_channels =
        anchors_per_cell * 4;

    const int class_channels =
        anchors_per_cell * 2;

    for (int y = 0; y < scale.height; ++y) {
        for (int x = 0; x < scale.width; ++x) {
            const int cell_index =
                y * scale.width + x;

            for (int anchor = 0;
                 anchor < anchors_per_cell;
                 ++anchor) {

                const int bbox_offset =
                    cell_index * bbox_channels +
                    anchor * 4;

                const int class_offset =
                    cell_index * class_channels +
                    anchor * 2;

                const float background_logit =
                    scale.class_data[class_offset];

                const float face_logit =
                    scale.class_data[class_offset + 1];

                const float face_score =
                    face_softmax(
                        background_logit,
                        face_logit
                    );

                if (face_score < score_threshold) {
                    continue;
                }

                // Raw predicted offsets:
                // [dx, dy, dw, dh]
                const float dx =
                    scale.bbox_data[bbox_offset];

                const float dy =
                    scale.bbox_data[bbox_offset + 1];

                const float dw =
                    scale.bbox_data[bbox_offset + 2];

                const float dh =
                    scale.bbox_data[bbox_offset + 3];

                const float minimum_size =
                    scale.min_sizes[anchor];

                // Normalized anchor dimensions.
                const float anchor_width =
                    minimum_size /
                    static_cast<float>(model_width);

                const float anchor_height =
                    minimum_size /
                    static_cast<float>(model_height);

                // The official Hailo postprocessor uses the
                // feature-map dimensions for anchor centers.
                const float anchor_center_x =
                    (static_cast<float>(x) + 0.5f) /
                    static_cast<float>(scale.width);

                const float anchor_center_y =
                    (static_cast<float>(y) + 0.5f) /
                    static_cast<float>(scale.height);

                // Decode using scale factors (10, 5):
                //
                // center = anchor center
                //          + offset / 10 * anchor size
                //
                // size = anchor size * exp(offset / 5)
                const float decoded_center_x =
                    anchor_center_x +
                    (dx / 10.0f) * anchor_width;

                const float decoded_center_y =
                    anchor_center_y +
                    (dy / 10.0f) * anchor_height;

                const float decoded_width =
                    anchor_width *
                    std::exp(
                        std::clamp(
                            dw / 5.0f,
                            -20.0f,
                            20.0f
                        )
                    );

                const float decoded_height =
                    anchor_height *
                    std::exp(
                        std::clamp(
                            dh / 5.0f,
                            -20.0f,
                            20.0f
                        )
                    );

                float x1 =
                    decoded_center_x -
                    decoded_width * 0.5f;

                float y1 =
                    decoded_center_y -
                    decoded_height * 0.5f;

                float x2 =
                    decoded_center_x +
                    decoded_width * 0.5f;

                float y2 =
                    decoded_center_y +
                    decoded_height * 0.5f;

                x1 = std::clamp(x1, 0.0f, 1.0f);
                y1 = std::clamp(y1, 0.0f, 1.0f);
                x2 = std::clamp(x2, 0.0f, 1.0f);
                y2 = std::clamp(y2, 0.0f, 1.0f);

                const int left =
                    static_cast<int>(
                        x1 * original_width
                    );

                const int top =
                    static_cast<int>(
                        y1 * original_height
                    );

                const int right =
                    static_cast<int>(
                        x2 * original_width
                    );

                const int bottom =
                    static_cast<int>(
                        y2 * original_height
                    );

                if (right <= left || bottom <= top) {
                    continue;
                }

                boxes.emplace_back(
                    left,
                    top,
                    right - left,
                    bottom - top
                );

                scores.push_back(face_score);
            }
        }
    }
}


int main(int argc, char **argv)
{
    const std::string hef_path =
        "/home/sv/Developer/camera/backend/"
        "hailo8l_models/lightface_slim.hef";

    const std::string image_path =
        argc >= 2
            ? argv[1]
            : "download_audio.jpg";

    const std::string output_path =
        argc >= 3
            ? argv[2]
            : "lightface_result.jpg";

    // Hailo's model configuration uses 0.1.
    // A higher value such as 0.5 produces fewer weak boxes.
    constexpr float SCORE_THRESHOLD = 0.50f;
    constexpr float NMS_THRESHOLD = 0.30f;

    // ---------------------------------------------------------
    // Load image
    // ---------------------------------------------------------

    cv::Mat original_image =
        cv::imread(image_path);

    if (original_image.empty()) {
        std::cerr
            << "Failed to load image: "
            << image_path
            << '\n';

        return 1;
    }

    cv::Mat annotated_image =
        original_image.clone();

    // ---------------------------------------------------------
    // Create device
    // ---------------------------------------------------------

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
    // Load model
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

    for (const auto &output_name :
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

        // Returns void in your installed HailoRT version.
        output_exp->set_format_type(
            HAILO_FORMAT_TYPE_FLOAT32
        );
    }

    // ---------------------------------------------------------
    // Configure model and bindings
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
    // Prepare input
    // ---------------------------------------------------------

    auto model_inputs =
        infer_model->inputs();

    if (model_inputs.size() != 1) {
        std::cerr
            << "Expected one input, found "
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
        rgb_image =
            rgb_image.clone();
    }

    std::vector<std::vector<uint8_t>>
        input_buffers(model_inputs.size());

    for (size_t i = 0;
         i < model_inputs.size();
         ++i) {

        const size_t expected_bytes =
            model_inputs[i].get_frame_size();

        const size_t actual_bytes =
            rgb_image.total() *
            rgb_image.elemSize();

        if (actual_bytes != expected_bytes) {
            std::cerr
                << "Input size mismatch. Image has "
                << actual_bytes
                << " bytes, but model expects "
                << expected_bytes
                << " bytes.\n";

            return 1;
        }

        input_buffers[i].resize(
            expected_bytes
        );

        std::memcpy(
            input_buffers[i].data(),
            rgb_image.data,
            expected_bytes
        );

        hailo_status status =
            bindings
                .input(model_inputs[i].name())
                ->set_buffer(
                    MemoryView(
                        input_buffers[i].data(),
                        input_buffers[i].size()
                    )
                );

        if (status != HAILO_SUCCESS) {
            std::cerr
                << "Failed to bind input: "
                << model_inputs[i].name()
                << ". Status: "
                << status
                << '\n';

            return 1;
        }
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
                << "Invalid FLOAT32 output size for "
                << model_outputs[i].name()
                << '\n';

            return 1;
        }

        output_buffers[i].resize(
            output_bytes / sizeof(float)
        );

        hailo_status status =
            bindings
                .output(model_outputs[i].name())
                ->set_buffer(
                    MemoryView(
                        output_buffers[i].data(),
                        output_bytes
                    )
                );

        if (status != HAILO_SUCCESS) {
            std::cerr
                << "Failed to bind output: "
                << model_outputs[i].name()
                << ". Status: "
                << status
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
    // Find an output by H × W × C
    // ---------------------------------------------------------

    auto find_output =
        [&](int height,
            int width,
            int channels) -> const float * {

        for (size_t i = 0;
             i < model_outputs.size();
             ++i) {

            const auto shape =
                model_outputs[i].shape();

            if (static_cast<int>(shape.height) ==
                    height &&
                static_cast<int>(shape.width) ==
                    width &&
                static_cast<int>(shape.features) ==
                    channels) {

                return output_buffers[i].data();
            }
        }

        return nullptr;
    };

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

    // ---------------------------------------------------------
    // Match the four classification/bbox pairs
    // ---------------------------------------------------------

    std::vector<ScaleOutput> scales = {
        {
            30,
            40,
            {10.0f, 16.0f, 24.0f},
            find_output(30, 40, 12),
            find_output(30, 40, 6)
        },
        {
            15,
            20,
            {32.0f, 48.0f},
            find_output(15, 20, 8),
            find_output(15, 20, 4)
        },
        {
            8,
            10,
            {64.0f, 96.0f},
            find_output(8, 10, 8),
            find_output(8, 10, 4)
        },
        {
            4,
            5,
            {128.0f, 192.0f, 256.0f},
            find_output(4, 5, 12),
            find_output(4, 5, 6)
        }
    };

    for (const auto &scale : scales) {
        if (scale.bbox_data == nullptr) {
            std::cerr
                << "Missing bbox tensor for scale "
                << scale.height << "x"
                << scale.width
                << '\n';

            return 1;
        }

        if (scale.class_data == nullptr) {
            std::cerr
                << "Missing classification tensor for scale "
                << scale.height << "x"
                << scale.width
                << '\n';

            return 1;
        }
    }

    // ---------------------------------------------------------
    // Decode predictions
    // ---------------------------------------------------------

    std::vector<cv::Rect> candidate_boxes;
    std::vector<float> candidate_scores;

    for (const auto &scale : scales) {
        decode_scale(
            scale,
            model_width,
            model_height,
            original_image.cols,
            original_image.rows,
            SCORE_THRESHOLD,
            candidate_boxes,
            candidate_scores
        );
    }

    std::cout
        << "Candidates above threshold: "
        << candidate_boxes.size()
        << '\n';

    // ---------------------------------------------------------
    // NMS
    // ---------------------------------------------------------

    std::vector<int> kept_indices;

    cv::dnn::NMSBoxes(
        candidate_boxes,
        candidate_scores,
        SCORE_THRESHOLD,
        NMS_THRESHOLD,
        kept_indices
    );

    // ---------------------------------------------------------
    // Draw retained detections
    // ---------------------------------------------------------

    for (int index : kept_indices) {
        const cv::Rect &box =
            candidate_boxes[index];

        const float score =
            candidate_scores[index];

        cv::rectangle(
            annotated_image,
            box,
            cv::Scalar(0, 255, 0),
            2
        );

        std::ostringstream text;

        text
            << "face "
            << std::fixed
            << std::setprecision(2)
            << score;

        int baseline = 0;

        const cv::Size text_size =
            cv::getTextSize(
                text.str(),
                cv::FONT_HERSHEY_SIMPLEX,
                0.55,
                2,
                &baseline
            );

        const int text_x =
            std::clamp(
                box.x,
                0,
                std::max(
                    0,
                    annotated_image.cols -
                        text_size.width - 8
                )
            );

        const int background_top =
            std::max(
                0,
                box.y - text_size.height - 10
            );

        cv::rectangle(
            annotated_image,
            cv::Rect(
                text_x,
                background_top,
                text_size.width + 8,
                text_size.height + 10
            ),
            cv::Scalar(0, 255, 0),
            cv::FILLED
        );

        cv::putText(
            annotated_image,
            text.str(),
            cv::Point(
                text_x + 4,
                background_top +
                    text_size.height + 3
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(0, 0, 0),
            2
        );
    }

    std::cout
        << "Faces after NMS: "
        << kept_indices.size()
        << '\n';

    // ---------------------------------------------------------
    // Save and display
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
        << "Saved result to: "
        << output_path
        << '\n';

    cv::imshow(
        "LightFace detections",
        annotated_image
    );

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}