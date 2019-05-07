cmake_minimum_required(VERSION 2.8)
project(resnet)
enable_language(CXX)
set(TF_SOURCE_DIR
        /home/stuart/tensorflow-master)
include_directories(
        ${TF_SOURCE_DIR}
        ${TF_SOURCE_DIR}/bazel-genfiles
        ${TF_SOURCE_DIR}/tensorflow/contrib/makefile/gen/protobuf/include
        ${TF_SOURCE_DIR}/tensorflow/contrib/makefile/downloads/eigen
        ${TF_SOURCE_DIR}/tensorflow/contrib/makefile/downloads/absl
        ${TF_SOURCE_DIR}/tensorflow/contrib/makefile/downloads/gemmlowp
)
set(TF_LIBS
        ${PROJECT_SOURCE_DIR}/lib/libtensorflow_cc.so
        ${PROJECT_SOURCE_DIR}/lib/libtensorflow_framework.so)
set(CMAKE_CXX_FLAGS -std=c++11)
add_executable(res_test main.cpp resnet_model.h resnet_model.cpp all_weights_loader.h all_weights_loader.cpp fpga_quantized_conv.cc fpga_quantized_conv.h fpga_conv_kernel.hpp)
target_link_libraries(res_test ${TF_LIBS})
