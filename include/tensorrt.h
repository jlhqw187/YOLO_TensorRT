# pragma once

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>
#include <assert.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "config.h"
#include "utils.h"

#define CUDA_CHECK(status)                                               \
  do {                                                                   \
    auto ret = (status);                                                 \
    if (ret != cudaSuccess) {                                            \
      std::cout << "CUDA failed with error code: " << ret                \
                << ", reason: " << cudaGetErrorString(ret) << std::endl; \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override{
    // suppress info-level messages
    if (severity == Severity::kINFO) std::cout << msg << std::endl;
  }
 } ;




extern Logger gLogger;
extern nvinfer1::IRuntime* runtime;
extern nvinfer1::ICudaEngine* engine;
extern nvinfer1::IExecutionContext* context;
extern cudaStream_t stream;

extern float* gpu_buffers[2];
extern float* cpu_output_buffer;


void serialize_engine(std::string &onnx_name, std::string &engine_name);

void deserialize_engine(std::string& engine_name, nvinfer1::IRuntime** runtime, nvinfer1::ICudaEngine** engine, nvinfer1::IExecutionContext** context);

void prepare_buffers(nvinfer1::ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer);

void release_resources();