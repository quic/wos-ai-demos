/*
**************************************************************************************************
* Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause-Clear
**************************************************************************************************
*/

#include "DataUtil.hpp"
#include "QnnTypeMacros.hpp"
#include <IOTensor.hpp>
#include <Logger.hpp>
#include <Models/HeadPoseModel.hpp>
#include <Models/Model.hpp>
#include <Models/ModelUtils.hpp>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

cv::Mat resize_and_center_crop(cv::Mat img, int target_size) {
  float aspect = img.cols / static_cast<float>(img.rows);
  cv::Mat img_resized;
  if (aspect > 1) {
    // landscape orientation - wide image
    int res = aspect * target_size;
    cv::resize(img, img_resized, cv::Size(res, target_size));
  } else if (aspect < 1) {
    // portrait orientation - tall image
    int res = target_size / aspect;
    cv::resize(img, img_resized, cv::Size(target_size, res));
  } else {
    cv::resize(img, img_resized, cv::Size(target_size, target_size));
  }

  // Crop the center of the image
  int height = img_resized.rows;
  int width = img_resized.cols;
  int start_row = (height - target_size) / 2;
  int start_col = (width - target_size) / 2;
  int end_row = start_row + target_size;
  int end_col = start_col + target_size;
  cv::Rect roi(start_col, start_row, end_col - start_col, end_row - start_row);
  cv::Mat img_cropped = img_resized(roi);

  return img_cropped;
}

cv::Mat normalise(cv::Mat img) {
  img.convertTo(img, CV_32F, 1 / 256.0);
  cv::Scalar mean(0.485, 0.456, 0.406);
  cv::Scalar std_dev(0.229, 0.224, 0.225);
  img = (img - mean) / std_dev;
  return img;
}

void HeadPoseModel::preProcess(PreProcessContext preContext) {
  int target_size = 224;
  cv::Mat img = cv::imread(preContext.inputFiles[preContext.fileIndex], -1);
  this->image =
      cv::imread(preContext.inputFiles[preContext.fileIndex], cv::IMREAD_COLOR);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  cv::Mat resized_image = resize_and_center_crop(img, target_size);

  int width = resized_image.cols;
  int height = resized_image.rows;
  int channel = resized_image.channels();

  std::memcpy(preContext.buffer,
              reinterpret_cast<uint8_t *>(resized_image.data),
              width * channel * height * sizeof(uint8_t));
}

std::string HeadPoseModel::getModelName() { return "headpose"; }

bool HeadPoseModel::postProcess(PostProcessContext postContext) {
  qnn::tools::iotensor::IOTensor *iotensor =
      (qnn::tools::iotensor::IOTensor *)postContext.iotensor;
  Qnn_Tensor_t *output = &postContext.outputs[0];
  std::vector<size_t> dims;
  iotensor->fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(output),
                     QNN_TENSOR_GET_RANK(output));
  float *floatBuffer = nullptr;
  iotensor->convertToFloat(&floatBuffer, output);
  model_utils::put_text_on_image(this->image, floatBuffer[0], floatBuffer[1],
                                 floatBuffer[2]);
  cv::Mat res_img = model_utils::draw_3d_axis(
      this->image, floatBuffer[0], floatBuffer[1], floatBuffer[2], 100);
  model_utils::save_image(res_img, postContext, getModelName());
  return true;
}
