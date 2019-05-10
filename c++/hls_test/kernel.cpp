#include "kernel.h"

void kernel(uint8 *input, int input_zero_point, uint8 *filter, int filter_zero_point, int *output, int stride,
            int filter_left_offset, int filter_top_offset) {

    uint8 input_buffer[IN_CHANNEL][IN_SIZE][IN_SIZE];
    uint8 filter_buffer[IN_CHANNEL][OUT_CHANNEL][F_SIZE][F_SIZE];
    int output_buffer[OUT_CHANNEL][OUT_SIZE][OUT_SIZE];

}
