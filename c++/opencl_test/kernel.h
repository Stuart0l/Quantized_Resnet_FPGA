//
// Created by luoxh on 19-5-9.
//

#ifndef HLS_TEST_KERNEL_H
#define HLS_TEST_KERNEL_H

#include <cstddef>
#include <cstring>

#define MAX_INOUT_SIZE 802816 //56*56*256
#define MAX_FILTER_SIZE 2359296 //3*3*512*512
// kernel size parameters
#define OUT_SIZE 14
#define IN_SIZE 58
#define F_SIZE 3
#define IN_CHANNEL 8
#define OUT_CHANNEL 32

typedef unsigned char uint8;
typedef int int32;

void store_output(int32 output_data[MAX_INOUT_SIZE], int output_buffer[OUT_SIZE][OUT_SIZE][OUT_CHANNEL], int out_y,
                  int out_x, int output_height, int output_width, int filter_count, int out_channel);

void
kernel(uint8 per_input_data[MAX_INOUT_SIZE], uint8 filter_data2[MAX_FILTER_SIZE], int per_output_data[MAX_INOUT_SIZE],
       const int kernel_param[13]);

#endif //HLS_TEST_KERNEL_H
