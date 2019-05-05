//
// Created by stuart on 19-4-25.
//

#include "fpga_quantized_conv.h"
using namespace tensorflow;
using namespace std;

void ConvFunctor(const quint8* input_data, int input_batches, int input_height, int input_width, int input_depth,
                 int input_offset, const quint8* filter_data, int filter_height, int filter_width, int filter_count,
                 int filter_offset, int stride, Padding padding, qint32* output_data, int output_height,
                 int output_width, int output_shift, int output_offset, int output_mult){

    const int32 highest = static_cast<int32>(Eigen::NumTraits<qint32>::highest());
    const int32 lowest = static_cast<int32>(Eigen::NumTraits<qint32>::lowest());

    const int32 rounding = (output_shift < 1) ? 0 : (1 << (output_shift - 1));

    int filter_left_offset;
    int filter_top_offset;
    if (padding == VALID) {
        filter_left_offset =
                ((output_width - 1) * stride + filter_width - input_width + 1) / 2;
        filter_top_offset =
                ((output_height - 1) * stride + filter_height - input_height + 1) / 2;
    } else {
        filter_left_offset =
                ((output_width - 1) * stride + filter_width - input_width) / 2;
        filter_top_offset =
                ((output_height - 1) * stride + filter_height - input_height) / 2;
    }

    // If we've got multiple images in our input, work through each of them.
    for (int batch = 0; batch < input_batches; ++batch) {
        // Walk through all the output image values, sliding the filter to
        // different
        // positions in the input.
        for (int out_y = 0; out_y < output_height; ++out_y) {
            for (int out_x = 0; out_x < output_width; ++out_x) {
                // Each filter kernel produces one output channel.
                for (int out_channel = 0; out_channel < filter_count; ++out_channel) {
                    // We're going to calculate a single output value, which means we
                    // need to multiply a three dimensional kernel of weights against
                    // the current location within the input image.
                    /*
                      *-------------------------------...
                      |\ ^
                      | \in_depth
                      |  \ v
                      |   *-------------------------------...
                      |   |            ^
                      |   |       in_y_origin
                      |   |            v   \
                      |   |<in_x_origin>*---*^
                      |   |            \|   |filter_height
                      .   |             *---*v
                      .   |             <--->
                          .         filter_width
                          .
                    */
                    const int in_x_origin = (out_x * stride) - filter_left_offset;
                    const int in_y_origin = (out_y * stride) - filter_top_offset;
                    int32 total = 0;
                    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                            for (int in_channel = 0; in_channel < input_depth;
                                 ++in_channel) {
                                const int in_x = in_x_origin + filter_x;
                                const int in_y = in_y_origin + filter_y;
                                int32 input_value;
                                // If the location is outside the bounds of the input image,
                                // use zero as a default value.
                                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                                    (in_y < input_height)) {
                                    const quint8 input_source_value =
                                            input_data[(batch * input_height * input_width *
                                                        input_depth) +
                                                       (in_y * input_width * input_depth) +
                                                       (in_x * input_depth) + in_channel];
                                    // We're promoting the T1 type to a higher bit depth here as
                                    // we do the subtraction.
                                    input_value =
                                            static_cast<int32>(input_source_value) - input_offset;
                                } else {
                                    input_value = 0;
                                }
                                const quint8 filter_source_value =
                                        filter_data[(filter_y * filter_width * input_depth *
                                                     filter_count) +
                                                    (filter_x * input_depth * filter_count) +
                                                    (in_channel * filter_count) + out_channel];
                                // Another promotion to 32 bit, as above.
                                const int32 filter_value =
                                        static_cast<int32>(filter_source_value) - filter_offset;
                                total += (input_value * filter_value);
                            }
                        }
                    }
                    // Here we're applying scale factors to compress the 32 bit
                    // accumulated total to a potentially lower bit depth.
                    const int32_t output =
                            ((((total + output_offset) * output_mult) + rounding) >>
                                                                                  output_shift);
                    // We need to saturate the results against the largest and smallest
                    // values that can be represented in this type.
                    const int32 top_clamped_output = std::min(output, highest);
                    const int32 clamped_output = std::max(top_clamped_output, lowest);
                    output_data[(batch * output_height * output_width * filter_count) +
                                (out_y * output_width * filter_count) +
                                (out_x * filter_count) + out_channel] = clamped_output;
                }
            }
        }
    }
}


FpgaQuantizedConv::FpgaQuantizedConv(const tensorflow::ClientSession& session, ::tensorflow::Output input_,
                                     ::tensorflow::Tensor filter, ::tensorflow::Output input_min,
                                     ::tensorflow::Output input_max, float min_filter,
                                     float max_filter, const std::vector<int>& strides,
                                     std::string padding) {

    vector<Output> last_outputs = {input_, input_min, input_max};
    vector<Tensor> input_tensors;
    TF_CHECK_OK(session.Run(last_outputs, &input_tensors));
    // [batch, in_height, in_width, channels]
    const Tensor &input = input_tensors[0];
    // [filter_height, filter_width, in_channels, out_channels]

    const float min_input = input_tensors[1].flat<float>()(0);
    const float max_input = input_tensors[2].flat<float>()(0);

    Padding padding_ = VALID;
    if(padding == "SAME")
        padding_ = SAME;

    const int32 offset_input =
            FloatToQuantizedUnclamped<quint8>(0.0f, min_input, max_input);
    const int32 offset_filter =
            FloatToQuantizedUnclamped<quint8>(0.0f, min_filter, max_filter);
    const int32 offset_output = 0;
    const int32 mult_output = 1;
    const int32 shift_output = 0;

    // The last dimension for input is in_depth. It must be the same as the
    // filter's in_depth.
    const int64 in_depth = input.dim_size(3);

    // The last dimension for filter is out_depth.
    const int64 out_depth = filter.dim_size(3);

    // The second dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int64 input_rows = input.dim_size(1);
    const int64 filter_rows = filter.dim_size(0);

    // The third dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int64 input_cols = input.dim_size(2);
    const int64 filter_cols = filter.dim_size(1);

    // The first dimension for input is batch.
    const int64 batch = input.dim_size(0);

    // For now we take the stride from the second dimension only (we
    // assume row = col stride, and do not support striding on the
    // batch or depth dimension).
    const int stride = strides[1];

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    GetWindowedOutputSize(input_rows, filter_rows, stride,
                                         padding_, &out_rows, &pad_rows);
    GetWindowedOutputSize(input_cols, filter_cols, stride,
                                         padding_, &out_cols, &pad_cols);
    CHECK_GT(batch, 0);
    CHECK_GT(out_rows, 0);
    CHECK_GT(out_cols, 0);
    CHECK_GT(out_depth, 0);
    const TensorShape out_shape({batch, out_rows, out_cols, out_depth});

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor new_tensor(DT_QINT32, out_shape);
    output = std::move(new_tensor);

    ConvFunctor(input.flat<quint8>().data(), batch, input_rows, input_cols, in_depth, offset_input,
                filter.flat<quint8>().data(), filter_rows, filter_cols, out_depth, offset_filter,
                stride, padding_, output.flat<qint32>().data(), out_rows, out_cols,
                shift_output, offset_output, mult_output);

    QuantizationRangeForMultiplication<quint8, quint8, qint32>(
            min_input, max_input, min_filter, max_filter, &min_output,
            &max_output);
}
