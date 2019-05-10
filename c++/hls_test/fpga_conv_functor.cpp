//
// Created by luoxh on 19-5-9.
//

#include "fpga_conv_functor.h"
#include "kernel.h"

void ConvFunctor(const uint8* input_data, int input_batches, int input_height, int input_width, int input_depth,
                 uint8 input_zero_point, const uint8* filter_data, int filter_height, int filter_width, int filter_count,
                 uint8 filter_zero_point, int stride, Padding padding, qint32* output_data, int output_height,
                 int output_width){

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
        for (int out_y = 0; out_y < output_height; out_y+=OUT_SIZE) {
            for (int out_x = 0; out_x < output_width; out_x+=OUT_SIZE) {
                for (int out_channel = 0; out_channel < filter_count; out_channel+=OUT_CHANNEL) {
                    //bool reset = true;
                    int output_buffer[OUT_SIZE][OUT_SIZE][OUT_CHANNEL] = {0};
                    for (int in_channel = 0; in_channel < input_depth; in_channel+=IN_CHANNEL) {

                        for(int buffer_out_y = 0; buffer_out_y < OUT_SIZE; ++buffer_out_y) {
                            for (int buffer_out_x = 0; buffer_out_x < OUT_SIZE; ++buffer_out_x) {
                                int real_out_y = out_y + buffer_out_y;
                                int real_out_x = out_x + buffer_out_x;
                                if(real_out_y < output_height && real_out_x < output_width) {
                                    const int in_x_origin = (real_out_x * stride) - filter_left_offset;
                                    const int in_y_origin = (real_out_y * stride) - filter_top_offset;
                                    for (int buffer_out_channel = 0; buffer_out_channel < OUT_CHANNEL; buffer_out_channel++) {
                                        int total = 0;
                                        int sum_input = 0, sum_filter = 0;
                                        int calc_count = 0;
                                        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                                            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                                                for (int buffer_in_channel = 0; buffer_in_channel < IN_CHANNEL; ++buffer_in_channel) {
                                                    int real_out_channel = out_channel + buffer_out_channel;
                                                    int real_in_channel = in_channel + buffer_in_channel;
                                                    if (real_in_channel < input_depth && real_out_channel < filter_count) {
                                                        const int in_x = in_x_origin + filter_x;
                                                        const int in_y = in_y_origin + filter_y;
                                                        uint8 input_source_value;
                                                        const uint8 filter_source_value =
                                                                filter_data[(filter_y * filter_width * input_depth *
                                                                             filter_count) +
                                                                            (filter_x * input_depth * filter_count) +
                                                                            (real_in_channel * filter_count) +
                                                                            real_out_channel];
                                                        if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                                                            (in_y < input_height)) {
                                                            input_source_value =
                                                                    input_data[(batch * input_height * input_width *
                                                                                input_depth) +
                                                                               (in_y * input_width * input_depth) +
                                                                               (in_x * input_depth) + real_in_channel];
                                                            ++calc_count;
                                                            sum_filter += filter_source_value;
                                                            total += (input_source_value * filter_source_value);
                                                            sum_input += input_source_value;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        output_buffer[buffer_out_y][buffer_out_x][buffer_out_channel] +=
                                                total - sum_input * filter_zero_point - sum_filter * input_zero_point +
                                                input_zero_point * filter_zero_point * calc_count;
                                    }
                                }
                            }
                        }
                    }
                    for(int buffer_out_y = 0; buffer_out_y < OUT_SIZE; ++buffer_out_y){
                        for(int buffer_out_x = 0; buffer_out_x < OUT_SIZE; ++buffer_out_x){
                            int real_out_y = out_y + buffer_out_y;
                            int real_out_x = out_x + buffer_out_x;
                            if(real_out_y < output_height && real_out_x < output_width)
                                for(int buffer_out_channel = 0; buffer_out_channel < OUT_CHANNEL; ++buffer_out_channel){
                                    int real_out_channel = out_channel + buffer_out_channel;
                                    if(real_out_channel < filter_count)
                                        output_data[(batch * output_height * output_width * filter_count) +
                                                    (real_out_y * output_width * filter_count) +
                                                    (real_out_x * filter_count) + real_out_channel] = output_buffer[buffer_out_y][buffer_out_x][buffer_out_channel];
                                }
                        }
                    }
                }
            }
        }
    }
}