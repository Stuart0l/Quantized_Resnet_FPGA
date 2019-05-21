//
// Created by stuart on 19-4-25.
//

#include "tensorflow/cc/ops/standard_ops.h"
#include "fpga_quantized_conv.h"
#include "fpga_conv_functor.h"
#include <iostream>

using namespace tensorflow;
using namespace std;

FpgaQuantizedConv::FpgaQuantizedConv(const tensorflow::ClientSession& session, ::tensorflow::Output input_,
                                     ::tensorflow::Tensor filter, ::tensorflow::Output input_min,
                                     ::tensorflow::Output input_max, float min_filter,
                                     float max_filter, const std::vector<int>& strides,
                                     std::string& padding, Timer& timer) {

    vector<Output> last_outputs = {input_, input_min, input_max};
    vector<Tensor> input_tensors;
    TF_CHECK_OK(session.Run(last_outputs, &input_tensors));
    timer.stop();
    // [batch, in_height, in_width, channels]
    const Tensor &input = input_tensors[0];
    // [filter_height, filter_width, in_channels, out_channels]

    const float min_input = input_tensors[1].flat<float>()(0);
    const float max_input = input_tensors[2].flat<float>()(0);

    Padding padding_ = VALID;
    if(padding == "SAME")
        padding_ = SAME;

    const int offset_input = FloatToQuantizedUnclamped<quint8>(0.0f, min_input, max_input); //zero point
    const int offset_filter = FloatToQuantizedUnclamped<quint8>(0.0f, min_filter, max_filter); //zero point

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
    Tensor new_tensor(DT_INT32, out_shape);
    output = std::move(new_tensor);

    ConvFunctor(input.flat<uint8>().data(), batch, input_rows, input_cols, in_depth, offset_input,
                filter.flat<uint8>().data(), filter_rows, filter_cols, out_depth, offset_filter,
                stride, padding_, output.flat<int32>().data(), out_rows, out_cols, timer);

    QuantizationRangeForMultiplication<quint8, quint8, qint32>(
            min_input, max_input, min_filter, max_filter, &min_output,
            &max_output);

    timer.start();
}
