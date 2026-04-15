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
#include <Models/MidasV2SmallModel.hpp>
#include <Models/Model.hpp>
#include <Models/ModelUtils.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#ifndef __hexagon__
#include "PAL/Directory.hpp"
#include "PAL/FileOp.hpp"
#include "PAL/Path.hpp"
#endif
#include "PAL/StringOp.hpp"

void MidasV2SmallModel::preProcess(PreProcessContext preContext) {
  const int width = (int)preContext.dims[2];
  const int height = (int)preContext.dims[1];
  const int channels = (int)preContext.dims[3];
  cv::Mat img = cv::imread(preContext.inputFiles[preContext.fileIndex]);
  if (img.empty()) {
    throw std::runtime_error("Could not load image at " +
                             preContext.inputFiles[preContext.fileIndex]);
  }
  this->org_height = img.rows;
  this->org_width = img.cols;

  img.convertTo(img, CV_16UC3, 255);
  // remove this line if you are sending the image data in same dimention what model is expecting.
  img = model_utils::resize_image(img, width, height);
  int dims_ = width * height * channels * sizeof(uint16_t); // Updated to use 'channels'
  std::memcpy(preContext.buffer, reinterpret_cast<uint8_t *>(img.data), dims_);
}

bool MidasV2SmallModel::postProcess(PostProcessContext postContext) {
  qnn::tools::iotensor::IOTensor *iotensor =
      (qnn::tools::iotensor::IOTensor *)postContext.iotensor;
  auto output = &postContext.outputs[0];
  std::vector<size_t> dims;
  iotensor->fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(output),
                     QNN_TENSOR_GET_RANK(output));
  float *f_buffer = nullptr;
  iotensor->convertToFloat(&f_buffer, output); 

  const int width = (int)dims[2];
  const int height = (int)dims[1];
  cv::Mat image(height, width, CV_32F, f_buffer);
  double minVal;
  double maxVal;
  cv::minMaxLoc(image, &minVal, &maxVal);
  if (maxVal - minVal > 1.0) {
    double max_val = 255;
    cv::subtract(image, cv::Scalar(minVal), image);
    cv::multiply(image, cv::Scalar(max_val), image);
    cv::divide(image, cv::Scalar(maxVal - minVal), image);
    image = cv::Scalar(255, 255, 255) - image;   // comment if you want to invert image in output.
  }

  // Resize to org image shape
  cv::resize(image, image,
      cv::Size(this->org_width, this->org_height));

  model_utils::save_image(image, postContext, getModelName());
  return true;
}

std::string MidasV2SmallModel::getModelName() { return "midas_v2"; }
