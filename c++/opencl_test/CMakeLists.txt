cmake_minimum_required(VERSION 2.8)
project(cl_test)
set(CMAKE_CXX_FLAGS "-std=c++11 -DOCL -g")
set(TF_SOURCE_DIR
        /home/luoxh/tensorflow)
set(OCL_HARNESS_SRC
        harness/ocl_src/CLKernel.cpp harness/ocl_src/CLKernel.h
        harness/ocl_src/CLMemObj.cpp harness/ocl_src/CLMemObj.h
        harness/ocl_src/CLWorld.cpp harness/ocl_src/CLWorld.h)
set(OPENCL_LIB_DIR
        /opt/Xilinx/SDx/2017.4.op/runtime/lib/x86_64)
set(OPENCL_INC_DIR
        /opt/Xilinx/SDx/2017.4.op/runtime/include/1_2)
include_directories(
        ${TF_SOURCE_DIR}
        ${TF_SOURCE_DIR}/bazel-genfiles
        ${TF_SOURCE_DIR}/tensorflow/contrib/makefile/gen/protobuf/include
        ${TF_SOURCE_DIR}/tensorflow/contrib/makefile/downloads/eigen
        ${TF_SOURCE_DIR}/tensorflow/contrib/makefile/downloads/absl
        ${TF_SOURCE_DIR}/tensorflow/contrib/makefile/downloads/gemmlowp
        ${OPENCL_INC_DIR}
)

set(TF_LIBS
        ${PROJECT_SOURCE_DIR}/../lib/libtensorflow_cc.so.1
        ${PROJECT_SOURCE_DIR}/../lib/libtensorflow_framework.so.1)
set(OPENCL_LIBS
        ${OPENCL_LIB_DIR}/libxilinxopencl.so)

set(HOST_SOURCE
        main.cpp resnet_model.h resnet_model.cpp
        all_weights_loader.h all_weights_loader.cpp
        fpga_quantized_conv.cpp fpga_quantized_conv.h
        fpga_conv_functor.cpp fpga_conv_functor.h)

add_executable(host
        ${OCL_HARNESS_SRC}
        utils.cpp utils.h)
target_link_libraries(host
        ${TF_LIBS}
        ${OPENCL_LIBS})
