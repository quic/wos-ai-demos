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
#include <Models/Model.hpp>
#include <Models/ModelUtils.hpp>
#include <Models/QuickSRNetModel.hpp>
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

void QuickSRNetModel::preProcess(PreProcessContext preContext) {
  std::string image_path = preContext.inputFiles[preContext.fileIndex];
  cv::Mat input_image = cv::imread(image_path);
  if (input_image.empty()) {
    throw std::runtime_error("Could not load image at " + image_path);
  }

  const int feed_width = (int)preContext.dims[2];
  const int feed_height = (int)preContext.dims[1];

  cv::Mat resized_image =
      model_utils::resizeImage(input_image, feed_width, feed_height);
  cv::Mat cliped_image = model_utils::clipImage(resized_image);

  int channels = cliped_image.channels();
  int size_of_image = feed_width * feed_height * channels * sizeof(uint8_t);
  std::memcpy(preContext.buffer, cliped_image.data, size_of_image);
}

bool QuickSRNetModel::postProcess(PostProcessContext postContext) {
  qnn::tools::iotensor::IOTensor *iotensor =
      (qnn::tools::iotensor::IOTensor *)postContext.iotensor;
  auto output = &postContext.outputs[0];
  uint8_t *buffer =
      reinterpret_cast<uint8_t *>(QNN_TENSOR_GET_CLIENT_BUF(output).data);
  std::vector<size_t> dims;
  iotensor->fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(output),
                     QNN_TENSOR_GET_RANK(output));
  const int width = (int)dims[2];
  const int height = (int)dims[1];
  const int channels = (int)dims[3];

  // Use the channels variable in the cv::Mat constructor
  int type = (channels == 1) ? CV_8UC1 : (channels == 3) ? CV_8UC3 : CV_8UC(channels);
  cv::Mat image(height, width, type, buffer);
  model_utils::save_image(image, postContext, getModelName());
  return true;
}

std::string QuickSRNetModel::getModelName() { return "quicksrnet"; }
