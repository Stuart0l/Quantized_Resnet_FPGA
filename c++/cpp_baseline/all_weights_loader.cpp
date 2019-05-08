//
// Created by stuart on 19-4-21.
//

#include "all_weights_loader.h"

all_weights_loader::all_weights_loader(string &path) {

    Env* env = Env::Default();
    TensorProto gamma, beta, mu, sigma, conv, dense, bias;
    Tensor scale, offset, mean, variance, filter, mat, y;

    ReadBinaryProto(env, path + "batchnorm/pb/scale_0.pb", &gamma);
    CHECK_OK(scale.FromProto(gamma));
    ReadBinaryProto(env, path + "batchnorm/pb/offset_0.pb", &beta);
    CHECK_OK(offset.FromProto(beta));
    ReadBinaryProto(env, path + "batchnorm/pb/mean_0.pb", &mu);
    CHECK_OK(mean.FromProto(mu));
    ReadBinaryProto(env, path + "batchnorm/pb/variance_0.pb", &sigma);
    CHECK_OK(variance.FromProto(sigma));
    ReadBinaryProto(env, path + "conv/pb/filter_0_quint8_const.pb", &conv);
    CHECK_OK(filter.FromProto(conv));

    params = {{{{scale, offset, mean, variance}}}};
    filters = {{{filter}}};
    min_filter = {{{min_inputs[0]}}};
    max_filter = {{{max_inputs[0]}}};
    int index = 1;

    for (int layer : block_size) {

        ReadBinaryProto(env, path + "batchnorm/pb/scale_" + to_string(index) + ".pb", &gamma);
        CHECK_OK(scale.FromProto(gamma));
        ReadBinaryProto(env, path + "batchnorm/pb/offset_" + to_string(index) + ".pb", &beta);
        CHECK_OK(offset.FromProto(beta));
        ReadBinaryProto(env, path + "batchnorm/pb/mean_" + to_string(index) + ".pb", &mu);
        CHECK_OK(mean.FromProto(mu));
        ReadBinaryProto(env, path + "batchnorm/pb/variance_" + to_string(index) + ".pb", &sigma);
        CHECK_OK(variance.FromProto(sigma));
        ReadBinaryProto(env, path + "conv/pb/filter_" + to_string(index) + "_quint8_const.pb", &conv);
        CHECK_OK(filter.FromProto(conv));

        vector<Tensor> shortcut_param{scale, offset, mean, variance};
        Tensor shortcut_filter(filter);
        float shortcut_min = min_inputs[index];
        float shortcut_max = max_inputs[index];
        index ++;

        vector<vector<vector<Tensor>>> layer_param;
        vector<vector<Tensor>> layer_filter;
        vector<vector<float>> layer_min, layer_max;
        for (int block = 0; block < layer; block++){
            vector<vector<Tensor>> block_param;
            vector<Tensor> block_filter;
            vector<float> block_min, block_max;
            for (int i = 0; i < 3; i++){

                ReadBinaryProto(env, path + "batchnorm/pb/scale_" + to_string(index) + ".pb", &gamma);
                CHECK_OK(scale.FromProto(gamma));
                ReadBinaryProto(env, path + "batchnorm/pb/offset_" + to_string(index) + ".pb", &beta);
                CHECK_OK(offset.FromProto(beta));
                ReadBinaryProto(env, path + "batchnorm/pb/mean_" + to_string(index) + ".pb", &mu);
                CHECK_OK(mean.FromProto(mu));
                ReadBinaryProto(env, path + "batchnorm/pb/variance_" + to_string(index) + ".pb", &sigma);
                CHECK_OK(variance.FromProto(sigma));
                ReadBinaryProto(env, path + "conv/pb/filter_" + to_string(index) + "_quint8_const.pb", &conv);
                CHECK_OK(filter.FromProto(conv));

                block_param.push_back({scale, offset, mean, variance});
                block_filter.push_back(filter);
                block_min.push_back(min_inputs[index]);
                block_max.push_back(max_inputs[index]);
                index ++;
            }
            if(block == 0) {
                block_param.push_back(shortcut_param);
                block_filter.push_back(shortcut_filter);
                block_min.push_back(shortcut_min);
                block_max.push_back(shortcut_max);
            }
            layer_param.push_back(block_param);
            layer_filter.push_back(block_filter);
            layer_min.push_back(block_min);
            layer_max.push_back(block_max);
        }
        params.push_back(layer_param);
        filters.push_back(layer_filter);
        min_filter.push_back(layer_min);
        max_filter.push_back(layer_max);
    }

    ReadBinaryProto(env, path + "Matmul.pb", &dense);
    CHECK_OK(mat.FromProto(dense));
    ReadBinaryProto(env, path + "bias.pb", &bias);
    CHECK_OK(y.FromProto(bias));

    filters.push_back({{mat, y}});
    min_filter.push_back({{min_inputs[index]}});
    max_filter.push_back({{max_inputs[index]}});
}
