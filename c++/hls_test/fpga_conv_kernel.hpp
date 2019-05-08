//
// Created by stuart on 19-5-6.
//

#ifndef RESNET_FPGA_CONV_KERNEL_HPP
#define RESNET_FPGA_CONV_KERNEL_HPP

// kernel size parameters
const int Ty_out = 56;
const int Tx_out = 56;
const int Tx_in = 56;
const int Ty_in = 56;
const int Tx_f = 3;
const int Ty_f = 3;
const int Tc_in = 4;
const int Tc_out = 4;

void conv_kernel(uint8_t input[], uint8_t filter[], int32_t output[]){
    uint8_t filter_buffer[Tc_in][Tc_out][Ty_f][Tx_f];
    uint8_t input_buffer[Ty_in][Tx_in][Tc_in];
    int32_t output_buffer[Ty_out][Tx_out][Tc_out];

}
#endif //RESNET_FPGA_CONV_KERNEL_HPP
