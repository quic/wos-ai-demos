/*
**************************************************************************************************
* Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause-Clear
**************************************************************************************************
*/

#include <Models/DMSHNModel.hpp>
#include <Models/DeepFillV2Model.hpp>
#include <Models/DeepLabV3Model.hpp>
#include <Models/DexiNedModel.hpp>
#include <Models/HRNetModel.hpp>
#include <Models/HeadPoseModel.hpp>
#include <Models/MidasV2SmallModel.hpp>
#include <Models/Model.hpp>
#include <Models/ModelFactory.hpp>
#include <Models/ModelUtils.hpp>
#include <Models/QuickSRNetModel.hpp>
#include <Models/ModelOpenPose.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#ifndef __hexagon__
#include "PAL/Directory.hpp"
#include "PAL/FileOp.hpp"
#include "PAL/Path.hpp"
#endif

#define M_PI 3.14159265358979323846

const float minPeaksDistance = 3.0f;
const cv::Vec3f meanPixel = cv::Vec3f::all(128);
const float midPointsScoreThreshold = 0.05f;
const float foundMidPointsRatioThreshold = 0.8f;
const float minSubsetScore = 0.2f;
const float confidenceThreshold = 0.1f;
int minJointsNumber = 3;


model_utils::Poses::Poses(std::vector<cv::Point2f>& poses, float scores) : poses(poses), scores(scores) {};
model_utils::Keypoint::Keypoint(float x, float y, float score, int id) : x(x), y(y), score(score), id(id) {};
model_utils::Peak::Peak(const int id, const cv::Point2f& pos, const float score) : id(id), pos(pos), score(score) {};
model_utils::HumanPoseByPeaksIndices::HumanPoseByPeaksIndices(const int keypointsNumber)
    : peaksIndices(std::vector<int>(keypointsNumber, -1)),
    nJoints(0),
    score(0.0f) {};

model_utils::TwoJointsConnection::TwoJointsConnection(const int firstJointIdx, const int secondJointIdx, const float score)
    : firstJointIdx(firstJointIdx),
    secondJointIdx(secondJointIdx),
    score(score) {}

ModelFactory *model_utils::getModelFactory() {
  ModelFactory *factory = new ModelFactory();
  factory->getMap()->insert(
      std::make_pair(DexiNedModel::getModelName(), &createT<DexiNedModel>));
  factory->getMap()->insert(
      std::make_pair(DMSHNModel::getModelName(), &createT<DMSHNModel>));
  factory->getMap()->insert(std::make_pair(QuickSRNetModel::getModelName(),
                                           &createT<QuickSRNetModel>));
  factory->getMap()->insert(
      std::make_pair(DeepLabV3Model::getModelName(), &createT<DeepLabV3Model>));
  factory->getMap()->insert(std::make_pair(MidasV2SmallModel::getModelName(),
                                           &createT<MidasV2SmallModel>));
  factory->getMap()->insert(
      std::make_pair(HRNetModel::getModelName(), &createT<HRNetModel>));
  factory->getMap()->insert(std::make_pair(DeepFillV2Model::getModelName(),
                                           &createT<DeepFillV2Model>));
  factory->getMap()->insert(
      std::make_pair(HeadPoseModel::getModelName(), &createT<HeadPoseModel>));
  factory->getMap()->insert(
      std::make_pair(OpenPoseModel::getModelName(), &createT<OpenPoseModel>));
  return factory;
}

cv::Mat model_utils::computeSigmoid(const cv::Mat &inputTensor) {
  cv::Mat outputTensor;
  cv::exp(-inputTensor, outputTensor);
  outputTensor = 1.0 / (1.0 + outputTensor);
  return outputTensor;
}

// Convert image from BGR to RGB
cv::Mat model_utils::convertToRGB(const cv::Mat &img) {
  cv::Mat rgb_img;
  cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
  return rgb_img;
}

// Resize image using OpenCV
cv::Mat model_utils::resizeImage(const cv::Mat &img, int width, int height) {
  cv::Mat resized_img;
  cv::resize(img, resized_img, cv::Size(width, height), 0, 0,
             cv::INTER_LANCZOS4);
  return resized_img;
}

// Convert image to float and scale values from [0, 255] to [0, 1]
cv::Mat model_utils::convertToFloatAndScale(const cv::Mat &img) {
  cv::Mat float_img;
  img.convertTo(float_img, CV_32F, 1.0 / 255.0);
  return float_img;
}

void model_utils::save_image(const cv::Mat &image,
                             PostProcessContext postContext,
                             std::string model_name) {
  std::string output_dir = postContext.outputPath + "/" + model_name;
  std::string file_path =
      output_dir + "/output_" + std::to_string(postContext.startIdx) + ".jpg";
  if (!::pal::FileOp::checkFileExists(output_dir) &&
      !pal::Directory::makePath(output_dir)) {
    std::cerr << "Could not create output directory: " << output_dir
              << std::endl;
  }
  if (!cv::imwrite(file_path, image)) {
    std::cerr << "Failed to save the image." << std::endl;
  }
}

cv::Mat model_utils::clipImage(const cv::Mat &img) {
  cv::min(img, 255, img); // Upper limit
  cv::max(img, 0, img);   // Lower limit
  return img;
}

std::vector<cv::Vec3b> label_map = {
    cv::Vec3b(0, 0, 0),       // background
    cv::Vec3b(127, 0, 0),     // aeroplane
    cv::Vec3b(0, 128, 0),     // bicycle
    cv::Vec3b(128, 128, 0),   // bird
    cv::Vec3b(0, 0, 128),     // boat
    cv::Vec3b(128, 0, 128),   // bottle
    cv::Vec3b(0, 128, 128),   // bus
    cv::Vec3b(126, 126, 126), // car
    cv::Vec3b(64, 0, 0),      // cat
    cv::Vec3b(192, 0, 0),     // chair
    cv::Vec3b(64, 128, 0),    // cow
    cv::Vec3b(192, 128, 0),   // dining table
    cv::Vec3b(64, 0, 128),    // dog
    cv::Vec3b(192, 0, 128),   // horse
    cv::Vec3b(64, 128, 128),  // motorbike
    cv::Vec3b(128, 128, 128), // person
    cv::Vec3b(0, 64, 0),      // potted plant
    cv::Vec3b(128, 64, 0),    // sheep
    cv::Vec3b(0, 192, 0),     // sofa
    cv::Vec3b(128, 192, 0),   // train
    cv::Vec3b(0, 64, 128)     // tv/monitor
};

std::vector<std::string> COCO_INSTANCE_CATEGORY_NAMES = {"__background__",
                                                         "person",
                                                         "bicycle",
                                                         "car",
                                                         "motorcycle",
                                                         "airplane",
                                                         "bus",
                                                         "train",
                                                         "truck",
                                                         "boat",
                                                         "traffic light",
                                                         "fire hydrant",
                                                         "N / A",
                                                         "stop sign",
                                                         "parking meter",
                                                         "bench",
                                                         "bird",
                                                         "cat",
                                                         "dog",
                                                         "horse",
                                                         "sheep",
                                                         "cow",
                                                         "elephant",
                                                         "bear",
                                                         "zebra",
                                                         "giraffe",
                                                         "N / A",
                                                         "backpack",
                                                         "umbrella",
                                                         "N / A",
                                                         "N / A",
                                                         "handbag",
                                                         "tie",
                                                         "suitcase",
                                                         "frisbee",
                                                         "skis",
                                                         "snowboard",
                                                         "sports ball",
                                                         "kite",
                                                         "baseball bat",
                                                         "baseball glove",
                                                         "skateboard",
                                                         "surfboard",
                                                         "tennis racket",
                                                         "bottle",
                                                         "N / A",
                                                         "wine glass",
                                                         "cup",
                                                         "fork",
                                                         "knife",
                                                         "spoon",
                                                         "bowl",
                                                         "banana",
                                                         "apple",
                                                         "sandwich",
                                                         "orange",
                                                         "broccoli",
                                                         "carrot",
                                                         "hot dog",
                                                         "pizza",
                                                         "donut",
                                                         "cake",
                                                         "chair",
                                                         "couch",
                                                         "potted plant",
                                                         "bed",
                                                         "N / A",
                                                         "dining table",
                                                         "N / A",
                                                         "N / A",
                                                         "toilet",
                                                         "N / A",
                                                         "tv",
                                                         "laptop",
                                                         "mouse",
                                                         "remote",
                                                         "keyboard",
                                                         "cell phone",
                                                         "microwave",
                                                         "oven",
                                                         "toaster",
                                                         "sink",
                                                         "refrigerator",
                                                         "N / A",
                                                         "book",
                                                         "clock",
                                                         "vase",
                                                         "scissors",
                                                         "teddy bear",
                                                         "hair drier",
                                                         "toothbrush"};

const int SKELETON[17][2] = {{1, 3},   {1, 0},  {2, 4},   {2, 0},   {0, 5},
                             {0, 6},   {5, 7},  {7, 9},   {6, 8},   {8, 10},
                             {5, 11},  {6, 12}, {11, 12}, {11, 13}, {13, 15},
                             {12, 14}, {14, 16}};

vector<vector<int>> CocoColors = {
    {255, 0, 0},   {255, 85, 0},  {255, 170, 0}, {255, 255, 0}, {170, 255, 0},
    {85, 255, 0},  {0, 255, 0},   {0, 255, 85},  {0, 255, 170}, {0, 255, 255},
    {0, 170, 255}, {0, 85, 255},  {0, 0, 255},   {85, 0, 255},  {170, 0, 255},
    {255, 0, 255}, {255, 0, 170}, {255, 0, 85}};

cv::Mat model_utils::draw_segmentation_map(cv::Mat outputs) {
  cv::Mat red_map = cv::Mat::zeros(outputs.size(), CV_8UC1);
  cv::Mat green_map = cv::Mat::zeros(outputs.size(), CV_8UC1);
  cv::Mat blue_map = cv::Mat::zeros(outputs.size(), CV_8UC1);

  for (int label_num = 0; label_num < label_map.size(); ++label_num) {
    cv::Mat index;
    cv::compare(outputs, label_num, index, cv::CMP_EQ);
    index.convertTo(index, CV_8UC1);

    red_map.setTo(label_map[label_num][2], index);
    green_map.setTo(label_map[label_num][1], index);
    blue_map.setTo(label_map[label_num][0], index);
  }

  std::vector<cv::Mat> channels = {blue_map, green_map, red_map};
  cv::Mat segmentation_map;
  cv::merge(channels, segmentation_map);

  return segmentation_map;
}

cv::Mat model_utils::image_overlay(cv::Mat image, cv::Mat segmented_image) {
  int kernel_size = 15; // change based on blur required
  cv::Mat segmented_image_x;
  cv::compare(segmented_image, 128, segmented_image_x, cv::CMP_EQ);

  cv::Mat image_blur;
  cv::GaussianBlur(image, image_blur, cv::Size(kernel_size, kernel_size), 0);
  image.copyTo(image_blur, segmented_image_x);

  return image_blur;
}

float *model_utils::TransposeHWCToCHW(float *arr, int h, int w, int c) {
  float *transposed = new float[h * w * c];
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      for (int k = 0; k < c; k++) {
        transposed[k * h * w + i * w + j] = arr[i * w * c + j * c + k];
      }
    }
  }
  return transposed;
}

void model_utils::draw_pose(float *floatBuffer, cv::Mat img,
                            vector<size_t> dims) {
  // to draw the annotation on image.
  int frameWidth = img.cols;
  int frameHeight = img.rows;
  double thresh = 0.1;
  const int H = (int)dims[1];
  const int W = (int)dims[2];
  const int channel = (int)dims[3];
  cv::Mat output_(H, W, CV_32FC(channel), floatBuffer);
  vector<cv::Mat> channels(channel);
  split(output_, channels);
  vector<cv::Point> points(channel);
  for (int n = 0; n < channel; n++) {
    cv::Point2f p(-1, -1);
    cv::Point maxLoc;
    double prob;
    minMaxLoc(channels[n], 0, &prob, 0, &maxLoc);
    if (prob > thresh) {
      p = maxLoc;
      p.x *= (float)frameWidth / W;
      p.y *= (float)frameHeight / H;
    }
    points[n] = p;
  }

  int nPairs = sizeof(SKELETON) / sizeof(SKELETON[0]);

  for (int n = 0; n < nPairs; n++) {
    cv::Point2f partA = points[SKELETON[n][0]];
    cv::Point2f partB = points[SKELETON[n][1]];
    if (partA.x <= 0 || partA.y <= 0 || partB.x <= 0 || partB.y <= 0)
      continue;
    cv::Vec3b temp =
        cv::Vec3b(CocoColors[n][0], CocoColors[n][1], CocoColors[n][2]);
    line(img, partA, partB, temp, 8);
    circle(img, partA, 8, temp, -1);
    circle(img, partB, 8, temp, -1);
  }
}

int model_utils::constrain_to_multiple_of(int min_val = 0, int max_val = -1,
                                          float x = 1.0) {

  int y = static_cast<int>(std::round(x / 32) * 32);

  if (max_val != -1 && y > max_val) {
    y = static_cast<int>(std::floor(x / 32) * 32);
  }

  if (y < min_val) {
    y = static_cast<int>(std::ceil(x / 32) * 32);
  }

  return y;
}

cv::Size model_utils::get_size(int width, int height, int net_w, int net_h) {
  float scale_height = static_cast<float>(net_h) / height;
  float scale_width = static_cast<float>(net_w) / width;

  int new_height =
      model_utils::constrain_to_multiple_of(0, net_h, scale_height * height);
  int new_width =
      model_utils::constrain_to_multiple_of(0, net_w, scale_width * width);

  return cv::Size(new_width, new_height);
}

cv::Mat model_utils::resize_image(cv::Mat img, int net_w, int net_h) {
  cv::Size size = model_utils::get_size(img.cols, img.rows, net_w, net_h);
  cv::resize(img, img, size, 0, 0, cv::INTER_CUBIC);
  return img;
}

int findDot(string s) {
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == '.') {
      return i;
    }
  }
  return -1;
}

void model_utils::put_text_on_image(cv::Mat img, float a, float b, float c) {
  string s_yaw = "Yaw: " + to_string(a);
  string s_pitch = "Pitch: " + to_string(b);
  string s_roll = "Roll: " + to_string(c);
  string s_yaw_2 = s_yaw.substr(0, findDot(s_yaw) + 3);
  string s_pitch_2 = s_pitch.substr(0, findDot(s_pitch) + 3);
  string s_roll_2 = s_roll.substr(0, findDot(s_roll) + 3);
  cv::putText(img, s_yaw_2, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(img, s_pitch_2, cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(img, s_roll_2, cv::Point(20, 120), cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(0, 255, 0), 2);
}

std::vector<float> dot99(const std::vector<float>& roll, const std::vector<float>& pitch) {
    std::vector<float> res(9, 0.0f);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            res[i * 3 + j] = 0;
            for (int k = 0; k < 3; k++) {
                res[i * 3 + j] += roll[i * 3 + k] * pitch[k * 3 + j];
            }
        }
    }
    return res;
}

std::vector<float> dot93(const std::vector<float>& yaw, const std::vector<float>& arr) {
    std::vector<float> res(3, 0.0f);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            res[i] += yaw[i * 3 + j] * arr[j];
        }
    }
    return res;
}

cv::Mat model_utils::draw_3d_axis(cv::Mat img, float yaw, float pitch,
                                  float roll, int size) {
    float a_rad = yaw * M_PI / 180;
    float b_rad = pitch * M_PI / 180;
    float c_rad = roll * M_PI / 180;
    int length = size / 2;
    float r_yaw[9] = {cos(a_rad), -sin(a_rad), 0, sin(a_rad), cos(a_rad),
    0,          0,           0, 1};
    float r_pitch[9] = {cos(b_rad), 0,           sin(b_rad), 0,         1,
    0,          -sin(b_rad), 0,          cos(b_rad)};
    float r_roll[9] = {1,           0, 0,          0,         cos(c_rad),
    -sin(c_rad), 0, sin(c_rad), cos(c_rad)};
    float arr[3] = {length, 0, 0};

    std::vector<float> r_roll_vec(r_roll, r_roll + 9);
    std::vector<float> r_pitch_vec(r_pitch, r_pitch + 9);
    std::vector<float> r_yaw_vec(r_yaw, r_yaw + 9);
    std::vector<float> arr_vec(arr, arr + 3);
    
    std::vector<float> x_axis_vec = dot93(dot99(r_roll_vec, r_pitch_vec), dot93(r_yaw_vec, arr_vec));
    swap(arr_vec[0], arr_vec[1]);
    std::vector<float> y_axis_vec = dot93(dot99(r_roll_vec, r_pitch_vec), dot93(r_yaw_vec, arr_vec));
    swap(arr_vec[1], arr_vec[2]);
    std::vector<float> z_axis_vec = dot93(dot99(r_roll_vec, r_pitch_vec), dot93(r_yaw_vec, arr_vec));
    
    std::pair<int, int> center = {img.cols / 2, img.rows / 2};
    
    std::pair<int, int> x_end = {center.first + x_axis_vec[0], center.second - x_axis_vec[1]};
    std::pair<int, int> y_end = {center.first + y_axis_vec[0], center.second - y_axis_vec[1]};
    std::pair<int, int> z_end = {center.first + z_axis_vec[0], center.second - z_axis_vec[1]};
    
    cv::line(img, cv::Point(center.first, center.second),
             cv::Point(x_end.first, x_end.second), cv::Scalar(0, 0, 255), 1);
    cv::line(img, cv::Point(center.first, center.second),
             cv::Point(y_end.first, y_end.second), cv::Scalar(0, 255, 0), 1);
    cv::line(img, cv::Point(center.first, center.second),
             cv::Point(z_end.first, z_end.second), cv::Scalar(255, 0, 0), 1);
    return img;
}

class KeyPoints : public cv::ParallelLoopBody {
public:
    KeyPoints(const std::vector<cv::Mat>& heatMaps,
        float minPeaksDistance,
        std::vector<std::vector<model_utils::Peak>>& peaksFromHeatMap,
        float confidenceThreshold)
        : heatMaps(heatMaps),
        minPeaksDistance(minPeaksDistance),
        peaksFromHeatMap(peaksFromHeatMap),
        confidenceThreshold(confidenceThreshold) {}

    void operator()(const cv::Range& range) const override {
        for (int i = range.start; i < range.end; i++) {
            std::vector<cv::Point> peaks;
            const cv::Mat& heatMap = heatMaps[i];
            const float* heatMapData = heatMap.ptr<float>();
            size_t heatMapStep = heatMap.step1();
            for (int y = -1; y < heatMap.rows + 1; y++) {
                for (int x = -1; x < heatMap.cols + 1; x++) {
                    float val = 0;
                    if (x >= 0 && y >= 0 && x < heatMap.cols && y < heatMap.rows) {
                        val = heatMapData[y * heatMapStep + x];
                        val = val >= confidenceThreshold ? val : 0;
                    }

                    float left_val = 0;
                    if (y >= 0 && x < (heatMap.cols - 1) && y < heatMap.rows) {
                        left_val = heatMapData[y * heatMapStep + x + 1];
                        left_val = left_val >= confidenceThreshold ? left_val : 0;
                    }

                    float right_val = 0;
                    if (x > 0 && y >= 0 && y < heatMap.rows) {
                        right_val = heatMapData[y * heatMapStep + x - 1];
                        right_val = right_val >= confidenceThreshold ? right_val : 0;
                    }

                    float top_val = 0;
                    if (x >= 0 && x < heatMap.cols && y < (heatMap.rows - 1)) {
                        top_val = heatMapData[(y + 1) * heatMapStep + x];
                        top_val = top_val >= confidenceThreshold ? top_val : 0;
                    }

                    float bottom_val = 0;
                    if (x >= 0 && y > 0 && x < heatMap.cols) {
                        bottom_val = heatMapData[(y - 1) * heatMapStep + x];
                        bottom_val = bottom_val >= confidenceThreshold ? bottom_val : 0;
                    }

                    if ((val > left_val) && (val > right_val) && (val > top_val) && (val > bottom_val)) {
                        peaks.push_back(cv::Point(x, y));
                    }
                }
            }
            std::sort(peaks.begin(), peaks.end(), [](const cv::Point& a, const cv::Point& b) {
                return a.x < b.x;
                });
            std::vector<bool> isActualPeak(peaks.size(), true);
            int peakCounter = 0;
            std::vector<model_utils::Peak>& peaksWithScoreAndID = peaksFromHeatMap[i];
            for (size_t j = 0; j < peaks.size(); j++) { // Changed 'i' to 'j'
                if (isActualPeak[j]) {
                    for (size_t k = j + 1; k < peaks.size(); k++) { // Changed 'i' to 'j'
                        if (sqrt((peaks[j].x - peaks[k].x) * (peaks[j].x - peaks[k].x) +
                                 (peaks[j].y - peaks[k].y) * (peaks[j].y - peaks[k].y)) < minPeaksDistance) {
                            isActualPeak[k] = false;
                        }
                    }
                    peaksWithScoreAndID.push_back(model_utils::Peak(peakCounter++, peaks[j], heatMap.at<float>(peaks[j])));
                }
            }
        }
    }

private:
    const std::vector<cv::Mat>& heatMaps;
    float minPeaksDistance;
    std::vector<std::vector<model_utils::Peak>>& peaksFromHeatMap;
    float confidenceThreshold;
};

std::vector<model_utils::Poses> extractKeypoints(std::vector<cv::Mat> heatMaps, std::vector<cv::Mat> pafs)
{
    std::vector<std::vector<model_utils::Peak>> getValidPairs(heatMaps.size());
    KeyPoints KeyPoints(heatMaps, minPeaksDistance, getValidPairs, confidenceThreshold);
    cv::parallel_for_(cv::Range(0, static_cast<int>(heatMaps.size())), KeyPoints);
    int peaksBefore = 0;
    for (size_t heatmapId = 1; heatmapId < heatMaps.size(); heatmapId++) {
        peaksBefore += static_cast<int>(getValidPairs[heatmapId - 1].size());
        for (auto& peak : getValidPairs[heatmapId]) {
            peak.id += peaksBefore;
        }
    }

    static const std::pair<int, int> limbIdsHeatmap[] = { {2, 3},
                                                             {2, 6},
                                                             {3, 4},
                                                             {4, 5},
                                                             {6, 7},
                                                             {7, 8},
                                                             {2, 9},
                                                             {9, 10},
                                                             {10, 11},
                                                             {2, 12},
                                                             {12, 13},
                                                             {13, 14},
                                                             {2, 1},
                                                             {1, 15},
                                                             {15, 17},
                                                             {1, 16},
                                                             {16, 18},
                                                             {3, 17},
                                                             {6, 18} };
    static const std::pair<int, int> limbIdsPaf[] = { {31, 32},
                                                     {39, 40},
                                                     {33, 34},
                                                     {35, 36},
                                                     {41, 42},
                                                     {43, 44},
                                                     {19, 20},
                                                     {21, 22},
                                                     {23, 24},
                                                     {25, 26},
                                                     {27, 28},
                                                     {29, 30},
                                                     {47, 48},
                                                     {49, 50},
                                                     {53, 54},
                                                     {51, 52},
                                                     {55, 56},
                                                     {37, 38},
                                                     {45, 46} };

    const int keypointsNumber = 18;
    std::vector<model_utils::Peak> candidates;
    for (const auto& peaks : getValidPairs) {
        candidates.insert(candidates.end(), peaks.begin(), peaks.end());
    }
    std::vector<model_utils::HumanPoseByPeaksIndices> subset(0, model_utils::HumanPoseByPeaksIndices(18));
    for (size_t k = 0; k < model_utils::arraySize(limbIdsPaf); k++) {
        std::vector<model_utils::TwoJointsConnection> connections;
        const int mapIdxOffset = keypointsNumber + 1;
        std::pair<cv::Mat, cv::Mat> scoreMid = { pafs[limbIdsPaf[k].first - mapIdxOffset],
                                                pafs[limbIdsPaf[k].second - mapIdxOffset] };
        const int idxJointA = limbIdsHeatmap[k].first - 1;
        const int idxJointB = limbIdsHeatmap[k].second - 1;
        const std::vector<model_utils::Peak>& candA = getValidPairs[idxJointA];
        const std::vector<model_utils::Peak>& candB = getValidPairs[idxJointB];
        const size_t nJointsA = candA.size();
        const size_t nJointsB = candB.size();
        if (nJointsA == 0 && nJointsB == 0) {
            continue;
        }
        else if (nJointsA == 0) {
            for (size_t i = 0; i < nJointsB; i++) {
                int num = 0;
                for (size_t j = 0; j < subset.size(); j++) {
                    if (subset[j].peaksIndices[idxJointB] == candB[i].id) {
                        num++;
                        continue;
                    }
                }
                if (num == 0) {
                    model_utils::HumanPoseByPeaksIndices personKeypoints(keypointsNumber);
                    personKeypoints.peaksIndices[idxJointB] = candB[i].id;
                    personKeypoints.nJoints = 1;
                    personKeypoints.score = candB[i].score;
                    subset.push_back(personKeypoints);
                }
            }
            continue;
        }
        else if (nJointsB == 0) {
            for (size_t i = 0; i < nJointsA; i++) {
                int num = 0;
                for (size_t j = 0; j < subset.size(); j++) {
                    if (subset[j].peaksIndices[idxJointA] == candA[i].id) {
                        num++;
                        continue;
                    }
                }
                if (num == 0) {
                    model_utils::HumanPoseByPeaksIndices personKeypoints(keypointsNumber);
                    personKeypoints.peaksIndices[idxJointA] = candA[i].id;
                    personKeypoints.nJoints = 1;
                    personKeypoints.score = candA[i].score;
                    subset.push_back(personKeypoints);
                }
            }
            continue;
        }

        std::vector<model_utils::TwoJointsConnection> tempJointConnections;
        for (size_t i = 0; i < nJointsA; i++) {
            for (size_t j = 0; j < nJointsB; j++) {
                cv::Point2f pt = candA[i].pos * 0.5 + candB[j].pos * 0.5;
                cv::Point mid = cv::Point(cvRound(pt.x), cvRound(pt.y));
                cv::Point2f vec = candB[j].pos - candA[i].pos;
                double norm_vec = cv::norm(vec);
                if (norm_vec == 0) {
                    continue;
                }
                vec /= norm_vec;
                float score = vec.x * scoreMid.first.at<float>(mid) + vec.y * scoreMid.second.at<float>(mid);
                int height_n = pafs[0].rows / 2;
                float suc_ratio = 0.0f;
                float mid_score = 0.0f;
                const int mid_num = 10;
                const float scoreThreshold = -100.0f;
                if (score > scoreThreshold) {
                    float p_sum = 0;
                    int p_count = 0;
                    cv::Size2f step((candB[j].pos.x - candA[i].pos.x) / (mid_num - 1),
                        (candB[j].pos.y - candA[i].pos.y) / (mid_num - 1));
                    for (int n = 0; n < mid_num; n++) {
                        cv::Point midPoint(cvRound(candA[i].pos.x + n * step.width),
                            cvRound(candA[i].pos.y + n * step.height));
                        cv::Point2f pred(scoreMid.first.at<float>(midPoint), scoreMid.second.at<float>(midPoint));
                        score = vec.x * pred.x + vec.y * pred.y;
                        if (score > midPointsScoreThreshold) {
                            p_sum += score;
                            p_count++;
                        }
                    }
                    suc_ratio = static_cast<float>(p_count / mid_num);
                    float ratio = p_count > 0 ? p_sum / p_count : 0.0f;
                    mid_score = ratio + static_cast<float>(std::min(height_n / norm_vec - 1, 0.0));
                }
                if (mid_score > 0 && suc_ratio > foundMidPointsRatioThreshold) {
                    tempJointConnections.push_back(model_utils::TwoJointsConnection(i, j, mid_score));
                }
            }
        }
        if (!tempJointConnections.empty()) {
            std::sort(tempJointConnections.begin(),
                tempJointConnections.end(),
                [](const model_utils::TwoJointsConnection& a, const model_utils::TwoJointsConnection& b) {
                    return (a.score > b.score);
                });
        }
        size_t num_limbs = std::min(nJointsA, nJointsB);
        size_t cnt = 0;
        std::vector<int> occurA(nJointsA, 0);
        std::vector<int> occurB(nJointsB, 0);
        for (size_t row = 0; row < tempJointConnections.size(); row++) {
            if (cnt == num_limbs) {
                break;
            }
            const int& indexA = tempJointConnections[row].firstJointIdx;
            const int& indexB = tempJointConnections[row].secondJointIdx;
            const float& score = tempJointConnections[row].score;
            if (occurA[indexA] == 0 && occurB[indexB] == 0) {
                connections.push_back(model_utils::TwoJointsConnection(candA[indexA].id, candB[indexB].id, score));
                cnt++;
                occurA[indexA] = 1;
                occurB[indexB] = 1;
            }
        }
        if (connections.empty()) {
            continue;
        }

        bool extraJointConnections = (k == 17 || k == 18);
        if (k == 0) {
            subset = std::vector<model_utils::HumanPoseByPeaksIndices>(connections.size(), model_utils::HumanPoseByPeaksIndices(keypointsNumber));
            for (size_t i = 0; i < connections.size(); i++) {
                const int& indexA = connections[i].firstJointIdx;
                const int& indexB = connections[i].secondJointIdx;
                subset[i].peaksIndices[idxJointA] = indexA;
                subset[i].peaksIndices[idxJointB] = indexB;
                subset[i].nJoints = 2;
                subset[i].score = candidates[indexA].score + candidates[indexB].score + connections[i].score;
            }
        }
        else if (extraJointConnections) {
            for (size_t i = 0; i < connections.size(); i++) {
                const int& indexA = connections[i].firstJointIdx;
                const int& indexB = connections[i].secondJointIdx;
                for (size_t j = 0; j < subset.size(); j++) {
                    if (subset[j].peaksIndices[idxJointA] == indexA && subset[j].peaksIndices[idxJointB] == -1) {
                        subset[j].peaksIndices[idxJointB] = indexB;
                    }
                    else if (subset[j].peaksIndices[idxJointB] == indexB && subset[j].peaksIndices[idxJointA] == -1) {
                        subset[j].peaksIndices[idxJointA] = indexA;
                    }
                }
            }
            continue;
        }
        else {
            for (size_t i = 0; i < connections.size(); i++) {
                const int& indexA = connections[i].firstJointIdx;
                const int& indexB = connections[i].secondJointIdx;
                bool num = false;
                for (size_t j = 0; j < subset.size(); j++) {
                    if (subset[j].peaksIndices[idxJointA] == indexA) {
                        subset[j].peaksIndices[idxJointB] = indexB;
                        subset[j].nJoints++;
                        subset[j].score += candidates[indexB].score + connections[i].score;
                        num = true;
                    }
                }
                if (!num) {
                    model_utils::HumanPoseByPeaksIndices hpWithScore(keypointsNumber);
                    hpWithScore.peaksIndices[idxJointA] = indexA;
                    hpWithScore.peaksIndices[idxJointB] = indexB;
                    hpWithScore.nJoints = 2;
                    hpWithScore.score = candidates[indexA].score + candidates[indexB].score + connections[i].score;
                    subset.push_back(hpWithScore);
                }
            }
        }
    }
    std::vector<model_utils::Poses> poses;
    for (const auto& subsetI : subset) {
        if (subsetI.nJoints < minJointsNumber || subsetI.score / subsetI.nJoints < minSubsetScore) {
            continue;
        }
        int position = -1;
        model_utils::Poses pose{ std::vector<cv::Point2f>(19, cv::Point2f(-1.0f, -1.0f)),
                       subsetI.score * std::max(0, subsetI.nJoints - 1) };
        for (const auto& peakIdx : subsetI.peaksIndices) {
            position++;
            if (peakIdx >= 0) {
                pose.poses[position] = candidates[peakIdx].pos;
                pose.poses[position].x += 0.5;
                pose.poses[position].y += 0.5;
            }
        }
        poses.push_back(pose);
    }
    return poses;
}


void model_utils::extractPoses(float* heatBuffer, float* pafBuffer, cv::Mat img,
    vector<size_t> heat_dims, vector<size_t> paf_dims) {

    //Variables
    int frameWidth = img.cols;
    int frameHeight = img.rows;
    int upSampleRatio = 2;
    int heat_H = (int)heat_dims[1];
    int heat_W = (int)heat_dims[2];
    std::vector<std::pair<int, int>> connectionsNodes;
    const cv::Point2f blankKeys(-1.0f, -1.0f);

    // Creating CV::MAT from Buffer and up-sampling according to variable. 
    cv::Mat heat_mats(heat_H, heat_W, CV_32FC((int)heat_dims[3]), heatBuffer);
    cv::resize(heat_mats, heat_mats, cv::Size(heat_H * upSampleRatio, heat_W * upSampleRatio), 0, 0, cv::INTER_CUBIC);
    cv::Mat paf_mats((int)paf_dims[1], (int)paf_dims[2], CV_32FC((int)paf_dims[3]), pafBuffer);
    cv::resize(paf_mats, paf_mats, cv::Size((int)paf_dims[1] * upSampleRatio, (int)paf_dims[2] * upSampleRatio), 0, 0, cv::INTER_CUBIC);

    //Seperate all Channels 
    vector<cv::Mat> heat_mat((int)heat_dims[3]);
    vector<cv::Mat> paf_mat((int)paf_dims[3]);
    split(heat_mats, heat_mat);
    split(paf_mats, paf_mat);

    std::vector<model_utils::Poses> poses = extractKeypoints(heat_mat, paf_mat);

    // setting the points on frame aspect ratio
    float aspect = 1.0f;
    if (frameHeight > frameWidth)
    {
        aspect = frameHeight / frameWidth;
    }
    else {
        aspect = frameWidth / frameHeight;
    }

    // color code for circle
    static const cv::Scalar colors[] = { cv::Scalar(255, 0, 0),
        cv::Scalar(255, 85, 0), cv::Scalar(255, 170, 0), cv::Scalar(255, 255, 0), cv::Scalar(170, 255, 0), cv::Scalar(85, 255, 0),
        cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 85), cv::Scalar(0, 255, 170), cv::Scalar(0, 255, 255), cv::Scalar(0, 170, 255),
        cv::Scalar(0, 85, 255), cv::Scalar(0, 0, 255), cv::Scalar(85, 0, 255), cv::Scalar(170, 0, 255), cv::Scalar(255, 0, 255),
        cv::Scalar(255, 0, 170), cv::Scalar(255, 0, 85) };

    // Skeleton to connect the points
    static const std::pair<int, int> SKELETONS[] = { {1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6}, {6, 7}, {1, 8}, {8, 9},
                                                      {9, 10}, {1, 11}, {11, 12}, {12, 13}, {1, 0}, {0, 14}, {14, 16},
                                                      {0, 15}, {15, 17} };

    
    if (!poses.empty()) {
        connectionsNodes.insert(connectionsNodes.begin(), std::begin(SKELETONS), std::end(SKELETONS));
    }
    
    // Drawing Pose on Image
    for (auto& pose : poses) {
        for (auto& keypoint : pose.poses) {
            if (keypoint != cv::Point2f(-1, -1)) {
                keypoint.x *= (float)(frameWidth * aspect / (upSampleRatio * heat_W));
                keypoint.y *= (float)(frameHeight * aspect / (upSampleRatio * heat_H));
            }
        }
        for (size_t Idx = 0; Idx < pose.poses.size(); Idx++) {
            if (pose.poses[Idx] != blankKeys) {
                cv::circle(img, pose.poses[Idx], 8, colors[Idx], -1);
            }
        }
        for (const auto& connectionsNode : connectionsNodes) {
            std::pair<cv::Point2f, cv::Point2f> poseKeypoints(pose.poses[connectionsNode.first],
                pose.poses[connectionsNode.second]);
            if (poseKeypoints.first == blankKeys || poseKeypoints.second == blankKeys) {
                continue;
            }

            float meanX = (poseKeypoints.first.x + poseKeypoints.second.x) / 2;
            float meanY = (poseKeypoints.first.y + poseKeypoints.second.y) / 2;
            cv::Point diff = poseKeypoints.first - poseKeypoints.second;
            double length = std::sqrt(diff.x * diff.x + diff.y * diff.y);
            int angle = static_cast<int>(std::atan2(diff.y, diff.x) * 180 / CV_PI);
            std::vector<cv::Point> polygon;
            cv::ellipse2Poly(cv::Point2d(meanX, meanY), cv::Size2d(length / 2, 4), angle, 0, 360, 1, polygon);
            cv::fillConvexPoly(img, polygon, colors[connectionsNode.second]);
        }
    }
}