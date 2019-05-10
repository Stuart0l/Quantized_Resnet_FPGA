//
// Created by luoxh on 19-5-9.
//

#ifndef RESNET_FPGA_CONV_FUNCTOR_H
#define RESNET_FPGA_CONV_FUNCTOR_H


#include "tensorflow/core/util/padding.h"

using namespace tensorflow;

void ConvFunctor(const uint8* input_data, int input_batches, int input_height, int input_width, int input_depth,
                 uint8 input_zero_point, const uint8* filter_data, int filter_height, int filter_width, int filter_count,
                 uint8 filter_zero_point, int stride, Padding padding, qint32* output_data, int output_height,
                 int output_width);

#endif //RESNET_FPGA_CONV_FUNCTOR_H