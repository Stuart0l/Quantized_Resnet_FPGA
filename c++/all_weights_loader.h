//
// Created by stuart on 19-4-21.
//

#ifndef RESNET_ALL_WEIGHTS_LOADER_H
#define RESNET_ALL_WEIGHTS_LOADER_H

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include <vector>
#include <string>

#define CHECK_OK(val) \
if(val) std::cout << "ok" << std::endl

using namespace std;
using namespace tensorflow;

const vector<int> block_size{3, 4, 6, 3};

const vector<float> min_inputs = {
#include "../weights/conv/min.txt"
};
const vector<float> max_inputs = {
#include "../weights/conv/max.txt"
};

class all_weights_loader {
public:
    all_weights_loader(string &path);

    vector<vector<vector<vector<Tensor>>>> params;
    vector<vector<vector<Tensor>>> filters;
    vector<vector<vector<float>>> min_filter;
    vector<vector<vector<float>>> max_filter;
};


#endif //RESNET_ALL_WEIGHTS_LOADER_H
