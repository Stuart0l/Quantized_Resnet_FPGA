//
// Created by luoxh on 19-5-9.
//

#ifndef HLS_TEST_KERNEL_H
#define HLS_TEST_KERNEL_H

// kernel size parameters
#define OUT_SIZE 14
#define IN_SIZE 58
#define F_SIZE 3
#define IN_CHANNEL 4
#define OUT_CHANNEL 4

typedef unsigned char uint8;

void kernel(uint8 input[], int input_zero_point, uint8 filter[], int filter_zero_point, int output[], int stride,
            int filter_left_offset, int filter_top_offset);

#endif //HLS_TEST_KERNEL_H
