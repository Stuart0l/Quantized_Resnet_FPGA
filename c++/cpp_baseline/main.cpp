#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "resnet_model.h"
#include "all_weights_loader.h"
int main(){
    using namespace tensorflow;
    using namespace tensorflow::ops;
    Scope root = Scope::NewRootScope();
    ClientSession session(root);
    std::vector<Tensor> outputs;

    string path = "/home/luoxh/Desktop/resnet/weights/";
    TensorProto input;
    ReadBinaryProto(Env::Default(), path + "ILSVRC2012_val_00000001.pb", &input);
    Tensor input_image;
    CHECK_OK(input_image.FromProto(input));
    Output output;
    all_weights_loader loader(path);
    output = resnet50(session, root, input_image, loader.filters, loader.min_filter, loader.max_filter, loader.params);
    TF_CHECK_OK(session.Run({output}, &outputs));
    std::cout<<outputs[0].DebugString(outputs[0].NumElements());
    return 0;
}