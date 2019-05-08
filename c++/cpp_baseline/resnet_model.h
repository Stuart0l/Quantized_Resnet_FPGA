//
// Created by stuart on 19-4-18.
//

#ifndef RESNET_RESNET_MODEL_H
#define RESNET_RESNET_MODEL_H
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
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
    ConvPadding(const Scope &scope, Input input, Input filters,
                Input min_input, Input max_input,
                float min_filter, float max_filter,
                int kernel_size, int strides);
    Output output;
};

class BottleneckBlock{
public:
    BottleneckBlock(const Scope &scope, Input input, vector<Tensor> &filters,
                    Input min_input, Input max_input,
                    vector<float> &min_filter, vector<float> &max_filter,
                    vector<vector<Tensor>> &param,
                    bool projection_shortcut, int strides);

    Output output;
    Output output_min;
    Output output_max;
};

class BlockLayer{
public:
    BlockLayer(const Scope& scope, Input input, vector<vector<Tensor>> &filters,
               Input min_input, Input max_input,
               vector<vector<float>> &min_filter, vector<vector<float>> &max_filter,
               vector<vector<vector<Tensor>>> &param,
               int blocks, int strides);

    Output output;
    Output output_min;
    Output output_max;
};

Output resnet50(const ClientSession& session, const Scope &scope, const Input& input,
                vector<vector<vector<Tensor>>> &filters,
                vector<vector<vector<float>>> &min_filter,
                vector<vector<vector<float>>> &max_filter,
                vector<vector<vector<vector<Tensor>>>> &param);
#endif //RESNET_RESNET_MODEL_H