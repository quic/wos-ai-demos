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
#include <Models/DMSHNModel.hpp>
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

void DMSHNModel::preProcess(PreProcessContext preContext) {
  // Load image
  this->image = cv::imread(preContext.inputFiles[preContext.fileIndex]);
  if (this->image.empty()) {
    throw std::runtime_error("Could not load image at " +
                             preContext.inputFiles[preContext.fileIndex]);
  }

  // Preprocess image
  const int feed_width = (int)preContext.dims[2];
  const int feed_height = (int)preContext.dims[1];
  const int channels = (int)preContext.dims[3];

  cv::Mat rgb_image = model_utils::convertToRGB(this->image);
  cv::Mat resized_image =
      model_utils::resizeImage(rgb_image, feed_width, feed_height);

  // Copying the image to Buffer
  int size_of_image = feed_width * feed_height * channels * sizeof(uint8_t);
  std::memcpy(preContext.buffer, resized_image.data, size_of_image);
}

bool DMSHNModel::postProcess(PostProcessContext postContext) {
  qnn::tools::iotensor::IOTensor *iotensor =
      (qnn::tools::iotensor::IOTensor *)postContext.iotensor;
  for (size_t outputIdx = 0; outputIdx < postContext.numOutputs; outputIdx++) {
    auto output = &postContext.outputs[outputIdx];
    std::vector<size_t> dims;
    iotensor->fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(output),
                       QNN_TENSOR_GET_RANK(output));
    const int width = (int)dims[2];
    const int height = (int)dims[1];
    // Copy data from the output pointer to Mat image
    uint8_t *buffer =
        reinterpret_cast<uint8_t *>(QNN_TENSOR_GET_CLIENT_BUF(output).data);
    cv::Mat image(height, width, CV_8UC3, buffer);

    // Convert BGR to RGB
    cv::Mat image_rgb;
    cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);

    // Resize to org image shape
    cv::Mat image_resized;
    cv::resize(image_rgb, image_resized,
               cv::Size(this->image.cols, this->image.rows));

    model_utils::save_image(image_resized, postContext, getModelName());
  }
  return true;
}

std::string DMSHNModel::getModelName() { return "dmshn"; }
