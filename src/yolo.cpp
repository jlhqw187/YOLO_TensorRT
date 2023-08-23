#include "yolo.h"

float GetBoxIoU(const Object &a, const Object &b) {
  cv::Rect2f intersection = a.box & b.box;
  const float i = intersection.area();
  const float u = a.box.area() + b.box.area() - i;
  return (i / u);
}

void SoftNMS(std::vector<Object>* objects, float nms_thresh, float confidence_thresh) {
  // 按confidence排序
  std::sort(objects->begin(), objects->end(), [](const Object &a, const Object &b) {return a.confidence > b.confidence;});
  std::vector<Object> reserved_objects;
  
  while (!objects->empty()) {
    // 从objects中取出首位放入reserved_objects
    const auto obj = objects->front();
    reserved_objects.push_back(obj);
    objects->erase(objects->begin());
    // 与objects中剩下的计算iou，当ious大于thresh时，降低待选行列的得分，如果此时低于confidence的thresh时，删去
    for (auto iter = objects->begin(); iter != objects->end();) {
      const float iou = GetBoxIoU(obj, *iter);
      if (iou > nms_thresh) {
        const float weight = std::exp(-(iou * iou) / 0.5f);
        iter->confidence *= weight;
      }
      if (iter->confidence < confidence_thresh) {
        iter = objects->erase(iter);
      } else {
        ++iter;
      }
    }
  }
  objects->swap(reserved_objects);
}

Preprocess_Result preprocess(cv::Mat input_image){
  // preprocess:resize and mat2blob
  cv::Mat resize_image;
  Preprocess_Result result;
  const float ratio = std::min(kInputW / (input_image.cols * 1.0f), kInputH / (input_image.rows * 1.0f));
  const int border_width = input_image.cols * ratio;
  const int border_height = input_image.rows * ratio;
  const int x_offset = (kInputW - border_width) / 2;
  const int y_offset = (kInputH - border_height) / 2;
  cv::resize(input_image, resize_image, cv::Size(border_width, border_height));
  cv::copyMakeBorder(resize_image, resize_image, y_offset, y_offset, x_offset,
                     x_offset, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
  cv::cvtColor(resize_image, resize_image, cv::COLOR_BGR2RGB);
  const int channels = resize_image.channels();
  const int width = resize_image.cols;
  const int height = resize_image.rows;
  float* inputBlob = new float[kInputW * kInputH * 3];

  for (int c = 0; c < channels; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        inputBlob[c * width * height + h * width + w] = resize_image.at<cv::Vec3b>(h, w)[c] / 255.0f;
      }
    }
  }
  result.height_scale = ratio;
  result.width_scale = ratio;
  result.x_offset = x_offset;
  result.y_offset = y_offset;
  result.inputBlob = inputBlob;
  return result;
}







float* infer(float* inputBlob){
  // std::vector<float *> input_data{};
  // std::vector<const float *> output_data;
  // 推理并记录时间
  auto start = std::chrono::steady_clock::now();

  // std::cout << "输入数据的大小" << input_data.size() << std::endl;
  // float* test = new float[3*640*640]();
  // float input_data[3 * 640 * 640];
  // preprocess(input_image, input_data); 
//   tensorrt_onnx_->Infer(input_data, output_data);
  for(std::size_t i = 0; i < 1; i++){
    const int inputIndex = 0;
    CUDA_CHECK(cudaMemcpyAsync(gpu_buffers[inputIndex], inputBlob, kBatchSize * 3 * kInputH * kInputW * sizeof(float), cudaMemcpyHostToDevice, stream));
    delete[] inputBlob;
  }

  // context->enqueueV2((void**)gpu_buffers, stream, nullptr);
  context->enqueue(1, (void**)gpu_buffers, stream, nullptr);
  // output_data.clear();

  // for (std::size_t i = 0; i < output_indexes_.size(); ++i) {
  //   const int index = output_indexes_.at(i);
  //   CUDA_CHECK(cudaMemcpyAsync(output_buffers_.at(i), buffers_[index],
  //                              1 * output_sizes_.at(i) * sizeof(float),
  //                              cudaMemcpyDeviceToHost, stream_));
  //   // 什么意思
  //   const float *output_buffer = output_buffers_.at(i);
  //   std::cout << "这是第" << i << "个buffer" << output_buffer << std::endl;
  //   output_data.push_back(output_buffer);
  // }
  for (std::size_t i = 0; i < 1; ++i) {
    const int outputIndex = 1;
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[outputIndex], 1 * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    // // 什么意思
    // const float *output_buffer = output_buffers_.at(i);
    // std::cout << "这是第" << i << "个buffer" << output_buffer << std::endl;
    // output_data.push_back(output_buffer);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto end = std::chrono::steady_clock::now();
  auto cost_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "Model inference cost time: " << cost_time << " ms" << std::endl;

  assert(cpu_output_buffer);
  return cpu_output_buffer;
}

void decode(float* infer_output, const Preprocess_Result& result, std::vector<Object> *objs){
  objs->clear();
  
  // 25200
  const int cell_num = 25200;
  // 85(bx, by, bw, bh, objectness, class)
  const int cell_size = 85;
  // const与非const的转换
  float *ptr = infer_output;
  // 对所有的输出结果解码
  for (int i = 0; i < cell_num; ++i, ptr += cell_size) {
    const float objectness = ptr[4];
    // 第一波筛选：objectness
    if (objectness >= kObjectnessThresh) {
      const int label = std::max_element(ptr + 5, ptr + cell_size) - (ptr + 5);
      // 计算方式
      const float confidence = ptr[5 + label] * objectness;
      if (confidence >= kConfThresh) {
        const float bx = ptr[0];
        const float by = ptr[1];
        const float bw = ptr[2];
        const float bh = ptr[3];

        Object obj;
        obj.box.x = (bx - bw * 0.5f - result.x_offset) / result.width_scale;
        obj.box.y = (by - bh * 0.5f - result.y_offset) / result.height_scale;
        obj.box.width = bw / result.width_scale;
        obj.box.height = bh / result.height_scale;
        obj.label = label;
        obj.confidence = confidence;
        objs->push_back(obj);
      }
    }
  } // i loop
}


void draw_objects(cv::Mat &image, const std::vector<Object> &objects) {
  for (const auto &obj : objects) {
    cv::Scalar box_color =
        cv::Scalar(color_list[obj.label][0], color_list[obj.label][1],
                   color_list[obj.label][2]);

    std::stringstream ss;
    ss << coco_names.at(obj.label) << ", " << std::setprecision(2) << std::fixed << obj.confidence * 100 << "%";
    const std::string text = ss.str();

    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, nullptr);
    const int x = obj.box.x;
    const int y = obj.box.y;
    
    // 画物体框和文字框
    cv::rectangle(image, obj.box, box_color * 255, 2);
    cv::rectangle(image, cv::Rect(x, y - text_size.height, text_size.width + 5, text_size.height + 5), box_color * 255, -1);

    cv::putText(image, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
  }
}

