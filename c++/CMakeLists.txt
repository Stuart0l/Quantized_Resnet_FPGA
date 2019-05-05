cmake_minimum_required(VERSION 2.8)
project(resnet)
enable_language(CXX)
include_directories(
        /home/stuart/tensorflow-master
        /home/stuart/tensorflow-master/bazel-genfiles
        /home/stuart/tensorflow-master/tensorflow/contrib/makefile/gen/protobuf/include
        /usr/local/include/eigen
        /home/stuart/tensorflow-master/tensorflow/contrib/makefile/downloads/absl
        /home/stuart/tensorflow-master/tensorflow/contrib/makefile/downloads/gemmlowp
)
set(TF_LIBS
        ${PROJECT_SOURCE_DIR}/lib/libtensorflow_cc.so
        ${PROJECT_SOURCE_DIR}/lib/libtensorflow_framework.so)
set(CMAKE_CXX_FLAGS -std=c++11)
add_executable(res_test main.cpp resnet_model.h resnet_model.cpp all_weights_loader.h all_weights_loader.cpp fpga_quantized_conv.cc fpga_quantized_conv.h)
target_link_libraries(res_test ${TF_LIBS})
