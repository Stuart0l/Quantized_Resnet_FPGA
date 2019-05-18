#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "resnet_model.h"
#include "all_weights_loader.h"
#include "utils.h"
#include "harness/ocl_src/CLWorld.h"

#define TARGET_DEVICE "xilinx:aws-vu9p-f1:4ddr-xpr-2pr:4.0"

int main(int argc, char **argv) {

    using namespace tensorflow;
    using namespace tensorflow::ops;
    using namespace rosetta;

    Scope root = Scope::NewRootScope();
    ClientSession session(root);
    std::vector<Tensor> outputs;

    std::string kernelFile;
    parse_sdaccel_command_line_args(argc, argv, kernelFile);
    CLWorld resnet_world(TARGET_DEVICE, CL_DEVICE_TYPE_ACCELERATOR);

    resnet_world.addProgram(kernelFile);

    CLKernel conv_kernel(resnet_world.getContext(), resnet_world.getProgram(), "fpgaconv", resnet_world.getDevice());

    int global_size[3] = {1, 1, 1};
    int local_size[3] = {1, 1, 1};
    conv_kernel.set_global(global_size);
    conv_kernel.set_local(local_size);

    resnet_world.addKernel(conv_kernel);

    string path = "/home/luoxh/Desktop/resnet/weights/";
    TensorProto input;
    ReadBinaryProto(Env::Default(), path + "ILSVRC2012_val_00000001.pb", &input);
    Tensor input_image;
    CHECK_OK(input_image.FromProto(input));
    Output output;
    all_weights_loader loader(path);
    output = resnet50(session, root, input_image, loader.filters, loader.min_filter, loader.max_filter, loader.params,
                      resnet_world);
    TF_CHECK_OK(session.Run({output}, &outputs));
    std::cout << outputs[0].DebugString(outputs[0].NumElements());

    resnet_world.releaseWorld();
    return 0;
}