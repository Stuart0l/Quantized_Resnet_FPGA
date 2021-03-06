//
// Created by luoxh on 19-5-9.
//

#include "fpga_conv_functor.h"
#include <fstream>
using namespace std;
string path = "/home/luoxh/Desktop/resnet";

void ConvFunctor(const uint8 *input_data, int input_batches, int input_height, int input_width, int input_depth,
                 uint8 input_zero_point, const uint8 *filter_data, int filter_height, int filter_width,
                 int filter_count,
                 uint8 filter_zero_point, int stride, tensorflow::Padding padding, int32 *output_data, int output_height,
                 int output_width, Timer& timer) {

    static int call = 0;
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
        /*
        ofstream input_writer, filter_writer, param_writer;
        input_writer.open(path + "/weights/emu/input_" + to_string(call), ios::out|ios::binary);
        filter_writer.open(path + "/weights/emu/filter_" + to_string(call), ios::out|ios::binary);
        param_writer.open(path + "/weights/emu/param_" + to_string(call), ios::out|ios::binary);

        if(input_writer.is_open())
            input_writer.write((char*)per_input_data, sizeof(uint8)*input_size);
        else
            cout << "failed" << endl;

        if(filter_writer.is_open())
            filter_writer.write((char*)filter_data2, sizeof(uint8)*filter_size);
        else
            cout<< "failed\n";

        if(param_writer.is_open())
            param_writer.write((char*)kernel_param, sizeof(int)*13);
        else
            cout<<"failed\n";

        input_writer.close();
        filter_writer.close();
        param_writer.close();
         */

        kernel(per_input_data, filter_data2, per_output_data, kernel_param);

        memcpy(output_data + batch * output_size, per_output_data, output_size * sizeof(int32));
    }
}