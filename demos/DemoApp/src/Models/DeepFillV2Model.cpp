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
#include <Models/DeepFillV2Model.hpp>
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
#include "fstream"

void DeepFillV2Model::preProcess(PreProcessContext preContext) {
  const int width = (int)preContext.dims[2];
  const int height = (int)preContext.dims[1];
  const int channels = (int)preContext.dims[3];
  int flag = (preContext.inputNodeIndex == 0 ? cv::COLOR_BGR2RGB
                                             : cv::IMREAD_GRAYSCALE);

  cv::Mat input_image =
      cv::imread(preContext.inputFiles[preContext.fileIndex], flag);
  if (input_image.empty()) {
    throw std::runtime_error("Could not load image at " +
                             preContext.inputFiles[preContext.fileIndex]);
  }
  // Resize Using OpenCV
  cv::Mat resized_img;
  cv::resize(input_image, resized_img, cv::Size(width, height), 0, 0,
             cv::INTER_LINEAR);

  // Store in int buffer
  int size_of_image = width * channels * height * sizeof(uint8_t);

  if (preContext.inputNodeIndex == 0) {
    this->input0 = resized_img;
  } else if (preContext.inputNodeIndex == 1) {
    this->input1 = resized_img;
  }
  std::memcpy(preContext.buffer, resized_img.data, size_of_image);
}

bool DeepFillV2Model::postProcess(PostProcessContext postContext) {
  qnn::tools::iotensor::IOTensor *iotensor =
      (qnn::tools::iotensor::IOTensor *)postContext.iotensor;
  auto output = &postContext.outputs[1];
  std::vector<size_t> dims;
  iotensor->fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(output),
                     QNN_TENSOR_GET_RANK(output));

  float *f_buffer = nullptr;
  iotensor->convertToFloat(&f_buffer, output);
  const int width = (int)dims[2];
  const int height = (int)dims[1];

  cv::Mat float_output(height, width, CV_32FC3, f_buffer);
  cv::Mat float_input0 = model_utils::convertToFloatAndScale(this->input0);
  cv::Mat float_input1_gray = model_utils::convertToFloatAndScale(this->input1);

  cv::Mat float_input1;
  std::vector<cv::Mat> channels(3, float_input1_gray);
  cv::merge(channels, float_input1);

  cv::Mat only_mask_output = float_output.mul(float_input1);
  cv::Mat inverse_mask_input;
  cv::subtract(cv::Scalar(1.0, 1.0, 1.0), float_input1, inverse_mask_input);
  cv::Mat input_image_without_mask = float_input0.mul(inverse_mask_input);
  cv::Mat output_mask_overlay_on_input;
  cv::add(input_image_without_mask, only_mask_output,
          output_mask_overlay_on_input);
  cv::Mat output_mask_overlay_on_input_int;
  output_mask_overlay_on_input.convertTo(output_mask_overlay_on_input_int,
                                         CV_8UC3, 255);
  cv::Mat output_mask_overlay_on_input_clipped =
      model_utils::clipImage(output_mask_overlay_on_input_int);

  model_utils::save_image(output_mask_overlay_on_input_clipped, postContext,
                          getModelName());
  return true;
}

std::string DeepFillV2Model::getModelName() { return "deepfillv2"; }

std::vector<std::string> DeepFillV2Model::getInputTensorNames() {
  return std::vector<std::string>{INPUT_0_NAME, INPUT_1_NAME};
}