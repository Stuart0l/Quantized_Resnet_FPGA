//
// Created by luoxh on 19-5-9.
//

#include "fpga_conv_functor.h"

void ConvFunctor(const uint8 *input_data, int input_batches, int input_height, int input_width, int input_depth,
                 uint8 input_zero_point, const uint8 *filter_data, int filter_height, int filter_width,
                 int filter_count,
                 uint8 filter_zero_point, int stride, tensorflow::Padding padding, int32 *output_data, int output_height,
                 int output_width, rosetta::CLWorld& world) {

    int filter_left_offset;
    int filter_top_offset;
    if (padding == tensorflow::VALID) {
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
        int input_size = input_height * input_width * input_depth;
        int filter_size = filter_height * filter_width * input_depth * filter_count;
        int output_size = output_height * output_width * filter_count;

        uint8 per_input_data[MAX_INOUT_SIZE];
        uint8 filter_data2[MAX_FILTER_SIZE];
        int per_output_data[MAX_INOUT_SIZE];

        memcpy(per_input_data, input_data + batch * input_size, input_size);
        memcpy(filter_data2, filter_data, filter_size);

        int kernel_param[13] = {
                output_height,
                output_width,
                input_depth,
                filter_count,
                input_height,
                input_width,
                filter_height,
                filter_width,
                input_zero_point,
                filter_zero_point,
                filter_left_offset,
                filter_top_offset,
                stride
        };

        rosetta::CLMemObj inputMem((void*)per_input_data, sizeof(uint8), MAX_INOUT_SIZE, CL_MEM_READ_ONLY);
        rosetta::CLMemObj filterMem((void*)filter_data2, sizeof(uint8), MAX_FILTER_SIZE, CL_MEM_READ_ONLY);
        rosetta::CLMemObj outputMem((void*)per_output_data, sizeof(int), MAX_INOUT_SIZE, CL_MEM_WRITE_ONLY);
        rosetta::CLMemObj paraMem((void*)kernel_param, sizeof(int), 13, CL_MEM_READ_ONLY);

        world.addMemObj(inputMem);
        world.addMemObj(filterMem);
        world.addMemObj(outputMem);
        world.addMemObj(paraMem);

        world.setMemKernelArg(0, 0, 0);
        world.setMemKernelArg(0, 1, 1);
        world.setMemKernelArg(0, 2, 2);
        world.setMemKernelArg(0, 3, 3);

        world.runKernels(false);

        world.readMemObj(3);

        world.removeMemObj();
        world.removeMemObj();
        world.removeMemObj();
        world.removeMemObj();

        /*kernel_part
        uint8 input_buffer[IN_SIZE][IN_SIZE][IN_CHANNEL] = {0};
        uint8 filter_buffer[F_SIZE][F_SIZE][IN_CHANNEL][OUT_CHANNEL] = {0};
        int output_buffer[OUT_SIZE][OUT_SIZE][OUT_CHANNEL] = {0};

        for (int out_y = 0; out_y < output_height; out_y += OUT_SIZE) {
            for (int out_x = 0; out_x < output_width; out_x += OUT_SIZE) {
                for (int out_channel = 0; out_channel < filter_count; out_channel += OUT_CHANNEL) {
                    for (int in_channel = 0; in_channel < input_depth; in_channel += IN_CHANNEL) {

                        load_input(per_input_data, input_buffer, input_height, input_width, input_depth, in_channel);

                        load_filter(filter_data2, filter_buffer, filter_height, filter_width, input_depth, filter_count, in_channel, out_channel);

                        cell_conv(input_buffer, input_zero_point,
                                  filter_buffer, filter_zero_point,
                                  output_buffer, stride,
                                  filter_left_offset, filter_top_offset, out_y, out_x, output_height, output_width,
                                  filter_height, filter_width, input_height, input_width, in_channel, out_channel,
                                  input_depth, filter_count);
                    }
                    store_output(per_output_data, output_buffer, out_y, out_x, output_height, output_width, filter_count, out_channel);
                }
            }
        }*/
        //kernel(per_input_data, filter_data2, per_output_data, kernel_param);

        memcpy(output_data + batch * output_size, per_output_data, output_size * sizeof(int32));
    }
}