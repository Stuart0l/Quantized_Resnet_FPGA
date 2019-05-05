//
// Created by stuart on 19-4-29.
//

#ifndef RESNET_FPGA_QUANTIZED_CONV_H
#define RESNET_FPGA_QUANTIZED_CONV_H

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/util/padding.h"
#include <vector>
#include <string>
#include <tensorflow/cc/client/client_session.h>


class FpgaQuantizedConv{
public:
    FpgaQuantizedConv(const tensorflow::ClientSession& session, ::tensorflow::Output input_,
                      ::tensorflow::Tensor filter, ::tensorflow::Output input_min,
                      ::tensorflow::Output input_max, float filter_min,
                      float filter_max, const std::vector<int>& strides,
                       std::string padding);

    ::tensorflow::Tensor output;
    float min_output;
    float max_output;
};


#endif //RESNET_FPGA_QUANTIZED_CONV_H
