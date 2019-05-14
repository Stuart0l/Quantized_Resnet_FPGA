#include "kernel.h"

void cell_conv(uint8 input_buffer[IN_SIZE][IN_SIZE][IN_CHANNEL], int input_zero_point,
               uint8 filter_buffer[F_SIZE][F_SIZE][IN_CHANNEL][OUT_CHANNEL], int filter_zero_point,
               int output_buffer[OUT_SIZE][OUT_SIZE][OUT_CHANNEL], int stride,
               int filter_left_offset, int filter_top_offset, int out_y, int out_x, int output_height, int output_width,
               int filter_height, int filter_width, int input_height, int input_width, int in_channel, int out_channel,
               int input_depth, int filter_count) {

    for (int buffer_out_y = 0; buffer_out_y < OUT_SIZE; ++buffer_out_y) {
        for (int buffer_out_x = 0; buffer_out_x < OUT_SIZE; ++buffer_out_x) {
            int real_out_y = out_y + buffer_out_y;
            int real_out_x = out_x + buffer_out_x;
            if (real_out_y < output_height && real_out_x < output_width) {
                const int in_x_origin = (real_out_x * stride) - filter_left_offset;
                const int in_y_origin = (real_out_y * stride) - filter_top_offset;
                for (int buffer_out_channel = 0;
                     buffer_out_channel < OUT_CHANNEL; buffer_out_channel++) {
                    int total = 0;
                    int sum_input = 0, sum_filter = 0;
                    int calc_count = 0;
                    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                            for (int buffer_in_channel = 0;
                                 buffer_in_channel < IN_CHANNEL; ++buffer_in_channel) {
                                int real_out_channel = out_channel + buffer_out_channel;
                                int real_in_channel = in_channel + buffer_in_channel;
                                if (real_in_channel < input_depth &&
                                    real_out_channel < filter_count) {
                                    const int in_x = in_x_origin + filter_x;
                                    const int in_y = in_y_origin + filter_y;
                                    uint8 input_source_value;
                                    const uint8 filter_source_value =
                                            filter_buffer[filter_y][filter_x][buffer_in_channel][buffer_out_channel];
                                    if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                                        (in_y < input_height)) {
                                        input_source_value =
                                                input_buffer[in_y][in_x][buffer_in_channel];
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

void
load_input(const uint8 input_data[MAX_INOUT_SIZE], uint8 input_buffer[IN_SIZE][IN_SIZE][IN_CHANNEL], int input_height,
           int input_width, int input_depth, int in_channel) {
    for (int buffer_in_y = 0; buffer_in_y < input_height; ++buffer_in_y) {
        for (int buffer_in_x = 0; buffer_in_x < input_width; ++buffer_in_x) {
            for (int buffer_in_channel = 0; buffer_in_channel < IN_CHANNEL; ++buffer_in_channel) {
                int real_in_channel = buffer_in_channel + in_channel;
                if (real_in_channel < input_depth) {
                    input_buffer[buffer_in_y][buffer_in_x][buffer_in_channel] =
                            input_data[(buffer_in_y * input_width * input_depth)
                                       + (buffer_in_x * input_depth)
                                       + real_in_channel];
                }
            }
        }
    }
}

void
load_filter(const uint8 filter_data[], uint8 filter_buffer[F_SIZE][F_SIZE][IN_CHANNEL][OUT_CHANNEL], int filter_height,
            int filter_width, int input_depth, int filter_count, int in_channel, int out_channel) {
    for (int buffer_f_y = 0; buffer_f_y < filter_height; ++buffer_f_y) {
        for (int buffer_f_x = 0; buffer_f_x < filter_width; ++buffer_f_x) {
            for (int buffer_in_channel = 0; buffer_in_channel < IN_CHANNEL; ++buffer_in_channel) {
                int real_in_channel = in_channel + buffer_in_channel;
                if (real_in_channel < input_depth) {
                    for (int buffer_out_channel = 0; buffer_out_channel < OUT_CHANNEL; ++buffer_out_channel) {
                        int real_out_channel = out_channel + buffer_out_channel;
                        if (real_out_channel < filter_count) {
                            filter_buffer[buffer_f_y][buffer_f_x][buffer_in_channel][buffer_out_channel] =
                                    filter_data[(buffer_f_y * filter_width * input_depth *
                                                 filter_count) +
                                                (buffer_f_x * input_depth * filter_count) +
                                                (real_in_channel * filter_count) +
                                                real_out_channel];
                        }
                    }
                }
            }
        }
    }
}

void
store_output(int output_data[MAX_INOUT_SIZE], int output_buffer[OUT_SIZE][OUT_SIZE][OUT_CHANNEL], int out_y, int out_x,
             int output_height, int output_width, int filter_count, int out_channel) {
    for (int buffer_out_y = 0; buffer_out_y < OUT_SIZE; ++buffer_out_y) {
        for (int buffer_out_x = 0; buffer_out_x < OUT_SIZE; ++buffer_out_x) {
            int real_out_y = out_y + buffer_out_y;
            int real_out_x = out_x + buffer_out_x;
            if (real_out_y < output_height && real_out_x < output_width) {
                for (int buffer_out_channel = 0;
                     buffer_out_channel < OUT_CHANNEL; ++buffer_out_channel) {
                    int real_out_channel = out_channel + buffer_out_channel;
                    if (real_out_channel < filter_count) {
                        output_data[(real_out_y * output_width * filter_count) +
                                    (real_out_x * filter_count) +
                                    real_out_channel] = output_buffer[buffer_out_y][buffer_out_x][buffer_out_channel];
                        output_buffer[buffer_out_y][buffer_out_x][buffer_out_channel] = 0;
                    }
                }
            }
        }
    }
}

void
kernel(uint8 per_input_data[MAX_INOUT_SIZE], uint8 filter_data2[MAX_FILTER_SIZE], int per_output_data[MAX_INOUT_SIZE],
       const int kernel_param[13]) {

    uint8 input_buffer[IN_SIZE][IN_SIZE][IN_CHANNEL] = {0};
    uint8 filter_buffer[F_SIZE][F_SIZE][IN_CHANNEL][OUT_CHANNEL] = {0};
    int output_buffer[OUT_SIZE][OUT_SIZE][OUT_CHANNEL] = {0};

    int output_height = kernel_param[0];
    int output_width = kernel_param[1];
    int input_depth = kernel_param[2];
    int filter_count = kernel_param[3];
    int input_height = kernel_param[4];
    int input_width = kernel_param[5];
    int filter_height = kernel_param[6];
    int filter_width = kernel_param[7];
    int input_zero_point = kernel_param[8];
    int filter_zero_point = kernel_param[9];
    int filter_left_offset = kernel_param[10];
    int filter_top_offset = kernel_param[11];
    int stride = kernel_param[12];

    for (int out_y = 0; out_y < output_height; out_y += OUT_SIZE) {
        for (int out_x = 0; out_x < output_width; out_x += OUT_SIZE) {
            for (int out_channel = 0; out_channel < filter_count; out_channel += OUT_CHANNEL) {
                for (int in_channel = 0; in_channel < input_depth; in_channel += IN_CHANNEL) {

                    load_input(per_input_data, input_buffer, input_height, input_width, input_depth, in_channel);

                    load_filter(filter_data2, filter_buffer, filter_height, filter_width, input_depth, filter_count,
                                in_channel, out_channel);

                    cell_conv(input_buffer, input_zero_point,
                              filter_buffer, filter_zero_point,
                              output_buffer, stride,
                              filter_left_offset, filter_top_offset, out_y, out_x, output_height, output_width,
                              filter_height, filter_width, input_height, input_width, in_channel, out_channel,
                              input_depth, filter_count);
                }
                store_output(per_output_data, output_buffer, out_y, out_x, output_height, output_width, filter_count,
                             out_channel);
            }
        }
    }

}
