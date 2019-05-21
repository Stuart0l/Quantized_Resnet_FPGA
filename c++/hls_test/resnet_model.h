//
// Created by stuart on 19-4-18.
//

#ifndef RESNET_RESNET_MODEL_H
#define RESNET_RESNET_MODEL_H
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "fpga_quantized_conv.h"
#include "timer.h"
#include <vector>

using namespace tensorflow;
using namespace std;

const vector<int> block_strides{1, 2, 2, 2};

class BatchNorm{
public:
    BatchNorm(const Scope &scope, Input input, vector<Tensor> &param);

    Output out;
};

class FixedPadding{
public:
    FixedPadding(const Scope &scope, Input input, int kernel_size);
    Output out;
};

class ConvPadding{
public:
    ConvPadding(const ClientSession& session, const Scope &scope, Input input, Tensor filters,
                Output min_input, Output max_input,
                float min_filter, float max_filter,
                int kernel_size, int strides, Timer& timer);
    Output output;
};

class BottleneckBlock{
public:
    BottleneckBlock(const ClientSession& session, const Scope &scope, Input input, vector<Tensor> &filters,
                    Output min_input, Output max_input,
                    vector<float> &min_filter, vector<float> &max_filter,
                    vector<vector<Tensor>> &param,
                    bool projection_shortcut, int strides, Timer& timer);

    Output output;
    Output output_min;
    Output output_max;
};

class BlockLayer{
public:
    BlockLayer(const ClientSession& session, const Scope& scope, Input input, vector<vector<Tensor>> &filters,
               Output min_input, Output max_input,
               vector<vector<float>> &min_filter, vector<vector<float>> &max_filter,
               vector<vector<vector<Tensor>>> &param,
               int blocks, int strides, Timer& timer);

    Output output;
    Output output_min;
    Output output_max;
};

Output resnet50(const ClientSession& session, const Scope &scope, const Input& input,
                vector<vector<vector<Tensor>>> &filters,
                vector<vector<vector<float>>> &min_filter,
                vector<vector<vector<float>>> &max_filter,
                vector<vector<vector<vector<Tensor>>>> &param, Timer& timer);
#endif //RESNET_RESNET_MODEL_H
