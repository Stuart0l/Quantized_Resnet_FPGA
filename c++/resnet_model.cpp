#include "all_weights_loader.h"
#include "resnet_model.h"
#include <string>

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

BatchNorm::BatchNorm(const Scope &scope, Input input, vector<Tensor> &param) {
    auto add = Add(scope, param[3], float(1e-3));
    auto rsqrt = Rsqrt(scope, add.z);
    auto mul = Mul(scope, rsqrt.y, param[0]);
    auto mul_1 = Mul(scope, mul.z, input);
    auto mul_2 = Mul(scope, mul.z, param[2]);
    auto sub = Sub(scope, param[1], mul_2.z);
    auto add_1 = Add(scope, mul_1.z, sub.z);
    out = add_1.z;
}

FixedPadding::FixedPadding(const Scope &scope, Input input, int kernel_size) {
    int pad_total = kernel_size - 1;
    int pad_beg = pad_total / 2;
    int pad_end = pad_total - pad_beg;
    out = Pad(scope, input, {{0,       0},
                             {pad_beg, pad_end},
                             {pad_beg, pad_end},
                             {0,       0}}).output;
}

ConvPadding::ConvPadding(const Scope &scope, Input input, Input filters,
                         Input min_input, Input max_input,
                         float min_filter, float max_filter,
                         int kernel_size, int strides) {

    string padding;
    if (strides == 1)
        padding = "SAME";
    else
        padding = "VALID";

    Output convout;
    Output convmin;
    Output convmax;
    if (strides > 1) {
        Dequantize Deq_4_pad(scope, input, min_input, max_input, Dequantize::Mode("MIN_FIRST"));
        FixedPadding pad(scope, Deq_4_pad.output, kernel_size);
        Reshape Resp(scope, pad.out, {-1});
        QuantizeV2 quant(scope, pad.out, Min(scope, Resp.output, 0).output, Max(scope, Resp.output, 0).output,
                         DT_QUINT8, QuantizeV2::Mode("MIN_FIRST"));
        QuantizedConv2D Qconv(scope, quant.output, filters, quant.output_min, quant.output_max, min_filter, max_filter,
                              {1, strides, strides, 1},
                              padding);
        convout = Qconv.output;
        convmin = Qconv.min_output;
        convmax = Qconv.max_output;
    } else {
        QuantizedConv2D Qconv(scope, input, filters, min_input, max_input, min_filter, max_filter,
                              {1, strides, strides, 1},
                              padding);
        convout = Qconv.output;
        convmin = Qconv.min_output;
        convmax = Qconv.max_output;
    }
    RequantizationRange ReqRange(scope, convout, convmin, convmax);
    Requantize Req(scope, convout, convmin, convmax, ReqRange.output_min, ReqRange.output_max,
                   DT_QUINT8);
    Dequantize Deq(scope, Req.output, Req.output_min, Req.output_max, Dequantize::Mode("MIN_FIRST"));
    output = Deq.output;
}

BottleneckBlock::BottleneckBlock(const Scope &scope, Input input, vector<Tensor> &filters,
                                 Input min_input, Input max_input,
                                 vector<float> &min_filter, vector<float> &max_filter,
                                 vector<vector<Tensor>> &param,
                                 bool projection_shortcut, int strides) {
    Output shortcut;

    if (projection_shortcut) {
        ConvPadding shortcut_conv(scope, input, filters[3], min_input, max_input, min_filter[3], max_filter[3],
                                  1, strides);
        shortcut = BatchNorm(scope, shortcut_conv.output, param[3]).out;
    }
    else
        shortcut = Dequantize(scope, input, min_input, max_input, Dequantize::Mode("MIN_FIRST")).output;

    //Conv 1
    ConvPadding Conv_1(scope, input, filters[0], min_input, max_input, min_filter[0], max_filter[0], 1, 1);
    BatchNorm Bn_1(scope, Conv_1.output, param[0]);
    Reshape Resp_1(scope, Bn_1.out, {-1});
    QuantizeV2 Quant_1(scope, Bn_1.out, Min(scope, Resp_1.output, 0).output, Max(scope, Resp_1.output, 0).output,
                       DT_QUINT8, QuantizeV2::Mode("MIN_FIRST"));
    QuantizedRelu QRelu_1(scope, Quant_1.output, Quant_1.output_min, Quant_1.output_max);

    //Conv 2
    ConvPadding Conv_2(scope, QRelu_1.activations, filters[1], QRelu_1.min_activations,
                       QRelu_1.max_activations,
                       min_filter[1], max_filter[1], 3, strides);
    BatchNorm Bn_2(scope, Conv_2.output, param[1]);
    Reshape Resp_2(scope, Bn_2.out, {-1});
    QuantizeV2 Quant_2(scope, Bn_2.out, Min(scope, Resp_2.output, 0).output, Max(scope, Resp_2.output, 0).output,
                       DT_QUINT8, QuantizeV2::Mode("MIN_FIRST"));
    QuantizedRelu QRelu_2(scope, Quant_2.output, Quant_2.output_min, Quant_2.output_max);

    //Conv 3
    ConvPadding Conv_3(scope, QRelu_2.activations, filters[2], QRelu_2.min_activations,
                       QRelu_2.max_activations,
                       min_filter[2], max_filter[2], 1, 1);
    BatchNorm Bn_3(scope, Conv_3.output, param[2]);
    Add add(scope, Bn_3.out, shortcut);
    Reshape Resp_3(scope, add.z, {-1});
    QuantizeV2 Quant_3(scope, add.z, Min(scope, Resp_3.output, 0).output, Max(scope, Resp_3.output, 0).output,
                       DT_QUINT8, QuantizeV2::Mode("MIN_FIRST"));
    QuantizedRelu QRelu_3(scope, Quant_3.output, Quant_3.output_min, Quant_3.output_max);

    output = QRelu_3.activations;
    output_min = QRelu_3.min_activations;
    output_max = QRelu_3.max_activations;
}

BlockLayer::BlockLayer(const Scope &scope, Input input, vector<vector<Tensor>> &filters,
                       Input min_input, Input max_input,
                       vector<vector<float>> &min_filter, vector<vector<float>> &max_filter,
                       vector<vector<vector<Tensor>>> &param,
                       int blocks, int strides) {

    BottleneckBlock block_0(scope, input, filters[0], min_input, max_input, min_filter[0], max_filter[0], param[0],
                            true,
                            strides);
    Output block_output = block_0.output;Output block_output_min = block_0.output_min;
    Output block_output_max = block_0.output_max;


    for (int i = 1; i < blocks; i++) {
        BottleneckBlock block_i(scope, block_output, filters[i], block_output_min, block_output_max,
                                min_filter[i], max_filter[i], param[i], false, 1);
        block_output = block_i.output;
        block_output_min = block_i.output_min;
        block_output_max = block_i.output_max;
    }

    output = block_output;
    output_min = block_output_min;
    output_max = block_output_max;
}

Output resnet50(const ClientSession &session, const Scope &scope, const Input &input,
                vector<vector<vector<Tensor>>> &filters,
                vector<vector<vector<float>>> &min_filter,
                vector<vector<vector<float>>> &max_filter,
                vector<vector<vector<vector<Tensor>>>> &param) {

    vector<Tensor> outputs;

    FixedPadding init_pad(scope, input, 7);
    Reshape init_Resp(scope, init_pad.out, {-1});
    QuantizeV2 init_quant(scope, init_pad.out, Min(scope, init_Resp.output, 0).output,
                          Max(scope, init_Resp.output, 0).output,
                          DT_QUINT8, QuantizeV2::Mode("MIN_FIRST"));
    QuantizedConv2D init_conv(scope, init_quant.output, filters[0][0][0], init_quant.output_min, init_quant.output_max,
                              min_filter[0][0][0], max_filter[0][0][0], {1, 2, 2, 1}, "VALID");
    RequantizationRange init_req_range(scope, init_conv.output, init_conv.min_output, init_conv.max_output);
    Requantize init_req(scope, init_conv.output, init_conv.min_output, init_conv.max_output,
                        init_req_range.output_min, init_req_range.output_max, DT_QUINT8);

    Dequantize init_deq(scope, init_req.output, init_req.output_min, init_req.output_max,
                        Dequantize::Mode("MIN_FIRST"));
    BatchNorm Bn(scope, init_deq.output, param[0][0][0]);
    Reshape Resp(scope, Bn.out, {-1});
    QuantizeV2 Quant(scope, Bn.out, Min(scope, Resp.output, 0).output, Max(scope, Resp.output, 0).output,
                     DT_QUINT8, QuantizeV2::Mode("MIN_FIRST"));
    QuantizedRelu QRelu(scope, Quant.output, Quant.output_min, Quant.output_max);
    QuantizedMaxPool QMaxp(scope, QRelu.activations, QRelu.min_activations, QRelu.max_activations,
                           {1, 3, 3, 1}, {1, 2, 2, 1}, "SAME");

    Output layer_output = QMaxp.output;
    Output layer_output_min = QMaxp.min_output;
    Output layer_output_max = QMaxp.max_output;

    for (int i = 0; i < 4; i++) {
        BlockLayer layer_i(scope, layer_output, filters[i + 1], layer_output_min, layer_output_max,
                           min_filter[i + 1], max_filter[i + 1], param[i + 1], block_size[i], block_strides[i]);
        layer_output = layer_i.output;
        layer_output_min = layer_i.output_min;
        layer_output_max = layer_i.output_max;
    }

    Dequantize Deq(scope, layer_output, layer_output_min, layer_output_max, Dequantize::Mode("MIN_FIRST"));
    Mean Red(scope, Deq.output, {1, 2}, Mean::KeepDims(true));
    Squeeze Squ(scope, Red.output, Squeeze::Axis({1, 2}));

    Reshape Resp1(scope, Squ.output, {-1});
    QuantizeV2 Quant1(scope, Squ.output, Min(scope, Resp1.output, 0).output, Max(scope, Resp1.output, 0).output,
                      DT_QUINT8, QuantizeV2::Mode("MIN_FIRST"));
    QuantizedMatMul dense(scope, {Quant1.output}, filters[5][0][0],
                          Quant1.output_min, Quant1.output_max, min_filter[5][0][0], max_filter[5][0][0]);
    RequantizationRange ReqRange(scope, dense.out, dense.min_out, dense.max_out);
    Requantize Req(scope, dense.out, dense.min_out, dense.max_out, ReqRange.output_min, ReqRange.output_max, DT_QUINT8);
    Dequantize Deq1(scope, Req.output, Req.output_min, Req.output_max, Dequantize::Mode("MIN_FIRST"));
    Add bias_add(scope, Deq1.output, filters[5][0][1]);

    return bias_add.z;
}
