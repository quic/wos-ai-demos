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
#include <Models/DexiNedModel.hpp>
#include <Models/Model.hpp>
#include <Models/ModelUtils.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

void DexiNedModel::preProcess(PreProcessContext preContext) {
  const int width = (int)preContext.dims[2];
  const int height = (int)preContext.dims[1];
  const int channels = (int)preContext.dims[3];
  cv::Mat input_image =
      cv::imread(preContext.inputFiles[preContext.fileIndex], cv::IMREAD_COLOR);
  cv::Mat resized_image;
  cv::resize(input_image, resized_image, cv::Size(width, height));
  std::memcpy(preContext.buffer,
              reinterpret_cast<uint8_t *>(resized_image.data),
              width * channels * height * sizeof(uint8_t));
}

bool DexiNedModel::postProcess(PostProcessContext postContext) {
  std::vector<cv::Mat> vec;
  qnn::tools::iotensor::IOTensor *iotensor =
      (qnn::tools::iotensor::IOTensor *)postContext.iotensor;
  for (size_t outputIdx = 0; outputIdx < postContext.numOutputs; outputIdx++) {
    QNN_DEBUG("Writing output for outputIdx: %d", outputIdx);
    Qnn_Tensor_t *output = &(postContext.outputs[outputIdx]);
    std::vector<size_t> dims;
    iotensor->fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(output),
                       QNN_TENSOR_GET_RANK(output));
    float *floatBuffer = nullptr;
    iotensor->convertToFloat(&floatBuffer, output);
    if (floatBuffer == nullptr) {
      QNN_ERROR("failure in convertToFloat");
      return false;
    }
    const int width = (int)dims[2];
    const int height = (int)dims[1];
    cv::Mat temp(height, width, CV_32F, floatBuffer);
    cv::Mat temp_norm;
    cv::Mat temp_sigmoid = model_utils::computeSigmoid(temp);
    cv::normalize(temp_sigmoid, temp_norm, 0, 255, cv::NORM_MINMAX);
    cv::Mat norm_cast;
    temp_norm.convertTo(norm_cast, CV_8U);
    cv::Mat temp_norm_comp;
    cv::bitwise_not(norm_cast, temp_norm_comp);
    vec.push_back(temp_norm_comp);
  }
  cv::Mat average = cv::Mat::zeros(vec[0].size(), CV_32F);
  for (int i = 0; i < (int)postContext.numOutputs; i++) {
    average += vec[i];
  }
  average /= postContext.numOutputs;
  model_utils::save_image(average, postContext, getModelName());
  return true;
}

std::string DexiNedModel::getModelName() { return "dexined"; }
