#pragma once

#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "tensorrt.h"

struct Object {
  // rec2f有x, y, width, height
  cv::Rect2f box;
  float confidence;
  int label;
};

struct Preprocess_Result{
  float width_scale;
  float height_scale;
  int x_offset;
  int y_offset;
  float* inputBlob;
};
// extern const std::vector<std::string> coco_name;
// extern const float color_list[80][3];
// inputBlob = new float[3 * kInputW * kInputH];

float GetBoxIoU(const Object &a, const Object &b);

void SoftNMS(std::vector<Object> *const objects, const float nms_thresh, const float confidence_thresh);

Preprocess_Result preprocess(cv::Mat input_image);

float* infer(float* inputBlob);

void decode(float* infer_output, const Preprocess_Result& result, std::vector<Object> *objs);

void draw_objects(cv::Mat &image, const std::vector<Object> &objects);
