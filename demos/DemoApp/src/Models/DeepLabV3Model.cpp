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
#include <Models/DeepLabV3Model.hpp>
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

void DeepLabV3Model::preProcess(PreProcessContext preContext) {
  const int width = (int)preContext.dims[2];
  const int height = (int)preContext.dims[1];
  const int channels = (int)preContext.dims[3];

  // Read Image
  input_image = cv::imread(preContext.inputFiles[preContext.fileIndex],
                           cv::COLOR_BGR2RGB);
  if (input_image.empty()) {
    throw std::runtime_error("Could not load image at " +
                             preContext.inputFiles[preContext.fileIndex]);
  }
  // Resize Image to model input dimension
  cv::Mat input_image_resized =
      model_utils::resizeImage(input_image, width, height);
  // Store in int buffer
  std::memcpy(preContext.buffer, input_image_resized.data,
              width * channels * height * sizeof(uint8_t));
}

bool DeepLabV3Model::postProcess(PostProcessContext postContext) {
  qnn::tools::iotensor::IOTensor *iotensor =
      (qnn::tools::iotensor::IOTensor *)postContext.iotensor;
  for (size_t outputIdx = 0; outputIdx < postContext.numOutputs; outputIdx++) {
    auto output = &postContext.outputs[outputIdx];
    // Output dimensions
    std::vector<size_t> dims;
    iotensor->fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(output),
                       QNN_TENSOR_GET_RANK(output));
    const int width = (int)dims[2];
    const int height = (int)dims[1];
    // buffer to CV::Mat
    uint32_t *buffer =
        reinterpret_cast<uint32_t *>(QNN_TENSOR_GET_CLIENT_BUF(output).data);
    cv::Mat mat(height, width, CV_32S, buffer);
    // Model output - Processing
    cv::Mat segmented_image = model_utils::draw_segmentation_map(mat);
    cv::Mat image_resized;
    // Read the Input image
    cv::resize(input_image, image_resized, cv::Size(width, height));

    // Overlay model output on image and resize image to org size
    cv::Mat final_image =
        model_utils::image_overlay(image_resized, segmented_image);
    cv::resize(final_image, final_image, input_image.size());
    model_utils::save_image(final_image, postContext, getModelName());
  }
  return true;
}

std::string DeepLabV3Model::getModelName() { return "deeplabv3"; }
