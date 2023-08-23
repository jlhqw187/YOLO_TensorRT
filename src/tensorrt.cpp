#include "tensorrt.h"




Logger gLogger;
nvinfer1::IRuntime* runtime = nullptr;
nvinfer1::ICudaEngine* engine = nullptr;
nvinfer1::IExecutionContext* context = nullptr;
float* gpu_buffers[2];
float* cpu_output_buffer = nullptr;
cudaStream_t stream;



void serialize_engine(std::string &onnx_name, std::string &engine_name) {
    // 从logger里Create builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    assert(builder != nullptr);

    // 从builder里创建config，config用来设置空间容量，及数据精度
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    builder->setMaxBatchSize(kBatchSize);//设置最大batchsize
    config->setMaxWorkspaceSize(8.0f * (1 << 30));// 32MB
#if defined(USE_FP16)
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
#elif defined(USE_INT8)
  std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
  assert(builder->platformHasFastInt8());
  config->setFlag(BuilderFlag::kINT8);
  Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, kClsInputW, kClsInputW, "./coco_calib/", "int8calib.table", kInputTensorName);
  config->setInt8Calibrator(calibrator);
#endif
    // 从network和config里创建engine
    //（step1:创建parser,network;step2:调用parser解析模型填充network;step3:标记网络输出;step4:创建engine）
    // rtx里面是自己搭建了一个yolov5然后载入wts
    
    //step3:创建network
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);  
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

    //step4:创建parser
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

    //step5:使用parser解析模型填充network
    // const char* onnx_filename="/fb_isa/workspace_zzh/git/onnx2eng/tensorrtx/yolov5/yolov5s.onnx";
    int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kINTERNAL_ERROR);

    parser->parseFromFile(onnx_name.c_str(), verbosity);

    // for (int i = 0; i < parser->getNbErrors(); ++i)
    // {
    //     std::cout << parser->getError(i)->desc() << std::endl;
    // }

    // // //step6:标记网络输出
    // onnx无需标记输出
    

    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine);
    
    // 序列化engine为ihostmemory，保存到文件
    nvinfer1::IHostMemory* serialized_engine = engine->serialize();
    assert(serialized_engine != nullptr);
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
      std::cerr << "Could not open plan output file" << std::endl;
      assert(false);
    }
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

    // 释放资源 builder config engine 及 ihostmemory
    engine->destroy();
    builder->destroy();
    config->destroy();
    serialized_engine->destroy();
}

void deserialize_engine(std::string& engine_name, nvinfer1::IRuntime** runtime, nvinfer1::ICudaEngine** engine, nvinfer1::IExecutionContext** context){

  std::ifstream file(engine_name, std::ios::binary);
  
  if (!file.good()) {
    std::cerr << "read " << engine_name << " error!" << std::endl;
    assert(false);
  }
  
  // 读取生成好的.engine文件
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  char* serialized_engine = new char[size];
  assert(serialized_engine);
  std::cout << "大小为" << size << std::endl;
  file.read(serialized_engine, size);
  file.close();
  
  std::cout << "load sucessfully " << engine_name << std::endl; 
  
  // 从logger里创建runtime
  
  *runtime = nvinfer1::createInferRuntime(gLogger);
  // nvinfer1::IRuntime* waitRuntime = nvinfer1::createInferRuntime(gLogger);runtime = &waitRuntime;
  assert(*runtime);

  // 从runtime里反序列化生成engine（指定engine数据，engine大小）
  *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
  // nvinfer1::ICudaEngine* waitEngine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
  // engine = &waitEngine;
  assert(*engine);

  assert((*engine)->getNbBindings() == 2);
  for (int bi = 0; bi < (*engine)->getNbBindings(); bi++) {
    if ((*engine)->bindingIsInput(bi) == true) printf("Binding %d (%s): Input.\n",  bi, (*engine)->getBindingName(bi));
    else printf("Binding %d (%s): Output.\n", bi, (*engine)->getBindingName(bi));
  }
  // 从engine里创建context（通过context管理中间激活值）
  *context = (*engine)->createExecutionContext();
  // nvinfer1::IExecutionContext* watiContext = (*engine)->createExecutionContext();
  // context = &watiContext;
  assert(*context);
  // 逐一析构
  delete[] serialized_engine;
}

void prepare_buffers(nvinfer1::ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer){


  // 根据engine和输入输出blob名字获取输入输出索引
  // for (int i = 0; i < engine->getNbBindings(); ++i) {
  //   if (engine->bindingIsInput(i))    input_indexes_.push_back(i);
  //   else  output_indexes_.push_back(i);
  // }
  // 两种方法获取输入输出索引
  const int inputIndex = engine->getBindingIndex(kInputTensorName);
  const int outputIndex = engine->getBindingIndex(kOutputTensorName);
  assert(inputIndex == 0);
  assert(outputIndex == 1);
  // 为输入输出开辟GPU显存
  CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * kOutputSize* sizeof(float)));

  *cpu_output_buffer = new float[kBatchSize * kOutputSize];
  CUDA_CHECK(cudaStreamCreate(&stream));

}

void release_resources(){
  // Release stream and buffers
  cudaStreamDestroy(stream);
  CUDA_CHECK(cudaFree(gpu_buffers[0]));
  CUDA_CHECK(cudaFree(gpu_buffers[1]));
  delete[] cpu_output_buffer;
  // Destroy the engine
  context->destroy();
  engine->destroy();
  runtime->destroy();
}
