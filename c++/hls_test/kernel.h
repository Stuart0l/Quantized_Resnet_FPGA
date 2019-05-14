//
// Created by luoxh on 19-5-9.
//

#ifndef HLS_TEST_KERNEL_H
#define HLS_TEST_KERNEL_H

#include <cstring>

#define MAX_INOUT_SIZE 802816 //56*56*256
#define MAX_FILTER_SIZE 2359296 //3*3*512*512
// kernel size parameters
#define OUT_SIZE 14
#define IN_SIZE 58
#define F_SIZE 3
#define IN_CHANNEL 4
#define OUT_CHANNEL 4

typedef unsigned char uint8;
typedef int int32;

void cell_conv(uint8 input[IN_SIZE][IN_SIZE][IN_CHANNEL], int input_zero_point,
               uint8 filter[F_SIZE][F_SIZE][IN_CHANNEL][OUT_CHANNEL], int filter_zero_point,
               int output[OUT_SIZE][OUT_SIZE][OUT_CHANNEL], int stride,
               int filter_left_offset, int filter_top_offset, int out_y, int out_x, int output_height, int output_width,
               int filter_height, int filter_width, int input_height, int input_width, int in_channel, int out_channel,
               int input_depth, int filter_count);

void load_input(const uint8 input_data[MAX_INOUT_SIZE], uint8 input_buffer[IN_SIZE][IN_SIZE][IN_CHANNEL],
                int input_height, int input_width, int input_depth, int in_channel);

void load_filter(const uint8 filter_data[MAX_FILTER_SIZE], uint8 filter_buffer[F_SIZE][F_SIZE][IN_CHANNEL][OUT_CHANNEL],
                 int filter_height, int filter_width, int input_depth, int filter_count, int in_channel,
                 int out_channel);

void store_output(int32 output_data[MAX_INOUT_SIZE], int output_buffer[OUT_SIZE][OUT_SIZE][OUT_CHANNEL], int out_y,
                  int out_x, int output_height, int output_width, int filter_count, int out_channel);

void
kernel(uint8 per_input_data[MAX_INOUT_SIZE], uint8 filter_data2[MAX_FILTER_SIZE], int per_output_data[MAX_INOUT_SIZE],
       const int kernel_param[13]);

#endif //HLS_TEST_KERNEL_H
