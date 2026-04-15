/*
**************************************************************************************************
* Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause-Clear
**************************************************************************************************
*/
#pragma once

#include <Models/Model.hpp>
#include <Models/ModelFactory.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

namespace model_utils {
ModelFactory *getModelFactory();

cv::Mat computeSigmoid(const cv::Mat &inputTensor);

// Convert image from BGR to RGB
cv::Mat convertToRGB(const cv::Mat &img);

// Resize image using OpenCV
cv::Mat resizeImage(const cv::Mat &img, int width, int height);

// Convert image to float and scale values from [0, 255] to [0, 1]
cv::Mat convertToFloatAndScale(const cv::Mat &img);

void save_image(const cv::Mat &image, PostProcessContext postContext,
                std::string model_name);

cv::Mat clipImage(const cv::Mat &img);

cv::Mat draw_segmentation_map(cv::Mat outputs);

cv::Mat image_overlay(cv::Mat image, cv::Mat segmented_image);

float *TransposeHWCToCHW(float *arr, int h, int w, int c);

void draw_pose(float *keypoint, cv::Mat img, vector<size_t> dims);

int constrain_to_multiple_of(int min_val, int max_val, float x);

cv::Size get_size(int width, int height, int net_w, int net_h);

cv::Mat resize_image(cv::Mat img, int net_w, int net_h);

void put_text_on_image(cv::Mat img, float a, float b, float c);

cv::Mat draw_3d_axis(cv::Mat img, float a, float b, float c, int size);

void extractPoses(float* heatBuffer, float* pafBuffer, cv::Mat img,
    vector<size_t> heatdims, vector<size_t> pafdims);

struct Poses {
    explicit Poses(std::vector<cv::Point2f>& poses = std::vector<cv::Point2f>(), float scores = 0.0f);
    std::vector<cv::Point2f> poses;
    float scores;
};

struct Peak {
    explicit Peak(const int id = -1, const cv::Point2f& pos = cv::Point2f(), const float score = 0.0f);
    int id;
    cv::Point2f pos;
    float score;
};

struct Keypoint {
    explicit Keypoint(float x = -1.0f, float y = -1.0f, float score = -1.0f, int id = 0);
    float x;
    float y;
    float score;
    int id;
};

struct HumanPoseByPeaksIndices {
    explicit HumanPoseByPeaksIndices(const int keypointsNumber);

    std::vector<int> peaksIndices;
    int nJoints;
    float score;
};

template <typename T, std::size_t N>
constexpr std::size_t arraySize(const T(&)[N]) noexcept {
    return N;
};

struct TwoJointsConnection {
    TwoJointsConnection(const int firstJointIdx, const int secondJointIdx, const float score);

    int firstJointIdx;
    int secondJointIdx;
    float score;
};

} // namespace model_utils