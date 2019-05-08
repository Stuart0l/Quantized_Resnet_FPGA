cmake_minimum_required(VERSION 2.8)
project(cpp_baseline)
enable_language(CXX)
set(TF_SOURCE_DIR
        /home/luoxh/tensorflow)
include_directories(
        ${TF_SOURCE_DIR}
        ${TF_SOURCE_DIR}/bazel-genfiles
        ${TF_SOURCE_DIR}/tensorflow/contrib/makefile/gen/protobuf/include
        ${TF_SOURCE_DIR}/tensorflow/contrib/makefile/downloads/eigen
        ${TF_SOURCE_DIR}/tensorflow/contrib/makefile/downloads/absl
        ${TF_SOURCE_DIR}/tensorflow/contrib/makefile/downloads/gemmlowp
)
set(TF_LIBS
        ${PROJECT_SOURCE_DIR}/../lib/libtensorflow_cc.so.1
        ${PROJECT_SOURCE_DIR}/../lib/libtensorflow_framework.so.1)
set(CMAKE_CXX_FLAGS -std=c++11)
add_executable(cpp_baseline main.cpp resnet_model.h resnet_model.cpp all_weights_loader.h all_weights_loader.cpp)
target_link_libraries(cpp_baseline ${TF_LIBS})
