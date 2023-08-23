#include <iostream>
#include <memory>
#include <string>
#include "utils.h"
#include "yolo.h"
#include "tensorrt.h"

// #include <NvInfer.h>
// #include <NvInferVersion.h>
// #include <NvOnnxConfig.h>
// #include <NvOnnxParser.h>
// #include <assert.h>
// #include <cuda_runtime_api.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>


int main(int argc, char **argv){
  // extern nvinfer1::ICudaEngine* engine;
  cudaSetDevice(kGpuId);
  if (argc < 4) {
  std::cout << "Usage: " << "-s" << " <path to onnx > <path to engine>\n";
  std::cout << "Usage: " << "-d" << " <path to engine > <path to image>\n";
  return -1;
  }
  if(strcmp(argv[1], "-s") == 0){
    std::string onnx_path(argv[2]);
    std::string engine_path(argv[3]);
    serialize_engine(onnx_path, engine_path);
    std::cout << "normally out!!!!!!!!!!!!!" << std::endl;
    return 0;
  }
  else{
    std::string engine_path(argv[2]);
    std::string image_path(argv[3]);
    deserialize_engine(engine_path, &runtime, &engine, &context);
    assert(engine);
    prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);
    cv::Mat input_image = cv::imread(image_path);
    Preprocess_Result result = preprocess(input_image);
    cpu_output_buffer = infer(result.inputBlob);
    std::vector<Object> objects;
    decode(cpu_output_buffer, result, &objects);
    SoftNMS(&objects, kNmsThresh, kConfThresh);
    draw_objects(input_image, objects);
    cv::imwrite("output.jpg", input_image);
    std::cout << "normally out!!!!!!!!!!!!!" << std::endl;
    release_resources();
    return 0;

  }
  
  // size_t find = onnx_path.find(".onnx");
  // if (find == std::string::npos) {
  //   std::cout << "The model file should be onnx format!\n";
  //   return false;
  // }
  // // engine和onnx会在一个目录下
  // std::string path = onnx_path.substr(0, find);
  // std::string engine_path = path + ".engine";
  // std::cout << argv[1] << std::endl;


}

