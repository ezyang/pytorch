cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# ---[ Project and semantic versioning.
project(Caffe2 CXX C)

option(USE_MKLDNN "Use MKLDNN" OFF)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

FIND_PACKAGE(MKLDNN)
