#pragma once
#include <vector>
#include <string>

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32

const static char* kInputTensorName = "images";
const static char* kOutputTensorName = "output0";

constexpr static int kNumClass = 80;
constexpr static int kBatchSize = 1;
constexpr static int kInputH = 640;
constexpr static int kInputW = 640;
constexpr static int kOutputSize = 25200 * 85;

constexpr static float kObjectnessThresh = 0.45f;
constexpr static float kConfThresh = 0.5f;
const static float kNmsThresh = 0.45f;


const static int kGpuId = 0;

static const std::vector<std::string> coco_names = {
    "person",        "bicycle",      "car",
    "motorcycle",    "airplane",     "bus",
    "train",         "truck",        "boat",
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",        "bird",
    "cat",           "dog",          "horse",
    "sheep",         "cow",          "elephant",
    "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",
    "kite",          "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket",
    "bottle",        "wine glass",   "cup",
    "fork",          "knife",        "spoon",
    "bowl",          "banana",       "apple",
    "sandwich",      "orange",       "broccoli",
    "carrot",        "hot dog",      "pizza",
    "donut",         "cake",         "chair",
    "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",
    "laptop",        "mouse",        "remote",
    "keyboard",      "cell phone",   "microwave",
    "oven",          "toaster",      "sink",
    "refrigerator",  "book",         "clock",
    "vase",          "scissors",     "teddy bear",
    "hair drier",    "toothbrush"};

const static float color_list[80][3] = {
    {0.000, 0.447, 0.741}, {0.850, 0.325, 0.098}, {0.929, 0.694, 0.125},
    {0.494, 0.184, 0.556}, {0.466, 0.674, 0.188}, {0.301, 0.745, 0.933},
    {0.635, 0.078, 0.184}, {0.300, 0.300, 0.300}, {0.600, 0.600, 0.600},
    {1.000, 0.000, 0.000}, {1.000, 0.500, 0.000}, {0.749, 0.749, 0.000},
    {0.000, 1.000, 0.000}, {0.000, 0.000, 1.000}, {0.667, 0.000, 1.000},
    {0.333, 0.333, 0.000}, {0.333, 0.667, 0.000}, {0.333, 1.000, 0.000},
    {0.667, 0.333, 0.000}, {0.667, 0.667, 0.000}, {0.667, 1.000, 0.000},
    {1.000, 0.333, 0.000}, {1.000, 0.667, 0.000}, {1.000, 1.000, 0.000},
    {0.000, 0.333, 0.500}, {0.000, 0.667, 0.500}, {0.000, 1.000, 0.500},
    {0.333, 0.000, 0.500}, {0.333, 0.333, 0.500}, {0.333, 0.667, 0.500},
    {0.333, 1.000, 0.500}, {0.667, 0.000, 0.500}, {0.667, 0.333, 0.500},
    {0.667, 0.667, 0.500}, {0.667, 1.000, 0.500}, {1.000, 0.000, 0.500},
    {1.000, 0.333, 0.500}, {1.000, 0.667, 0.500}, {1.000, 1.000, 0.500},
    {0.000, 0.333, 1.000}, {0.000, 0.667, 1.000}, {0.000, 1.000, 1.000},
    {0.333, 0.000, 1.000}, {0.333, 0.333, 1.000}, {0.333, 0.667, 1.000},
    {0.333, 1.000, 1.000}, {0.667, 0.000, 1.000}, {0.667, 0.333, 1.000},
    {0.667, 0.667, 1.000}, {0.667, 1.000, 1.000}, {1.000, 0.000, 1.000},
    {1.000, 0.333, 1.000}, {1.000, 0.667, 1.000}, {0.333, 0.000, 0.000},
    {0.500, 0.000, 0.000}, {0.667, 0.000, 0.000}, {0.833, 0.000, 0.000},
    {1.000, 0.000, 0.000}, {0.000, 0.167, 0.000}, {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000}, {0.000, 0.667, 0.000}, {0.000, 0.833, 0.000},
    {0.000, 1.000, 0.000}, {0.000, 0.000, 0.167}, {0.000, 0.000, 0.333},
    {0.000, 0.000, 0.500}, {0.000, 0.000, 0.667}, {0.000, 0.000, 0.833},
    {0.000, 0.000, 1.000}, {0.000, 0.000, 0.000}, {0.143, 0.143, 0.143},
    {0.286, 0.286, 0.286}, {0.429, 0.429, 0.429}, {0.571, 0.571, 0.571},
    {0.714, 0.714, 0.714}, {0.857, 0.857, 0.857}, {0.000, 0.447, 0.741},
    {0.314, 0.717, 0.741}, {0.50, 0.5, 0}};


