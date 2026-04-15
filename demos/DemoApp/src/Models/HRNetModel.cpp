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
#include <Models/HRNetModel.hpp>
#include <Models/Model.hpp>
#include <Models/ModelUtils.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

void HRNetModel::preProcess(PreProcessContext preContext) {

  const int width = (int)preContext.dims[2];
  const int height = (int)preContext.dims[1];
  const int channel = (int)preContext.dims[3];
  cv::Mat resized_img;
  image =
      cv::imread(preContext.inputFiles[preContext.fileIndex], cv::IMREAD_COLOR);
  cv::resize(this->image, resized_img, cv::Size((int)width, (int)height));
  std::memcpy(preContext.buffer, reinterpret_cast<uint8_t *>(resized_img.data),
              width * channel * height * sizeof(uint8_t));
}

bool HRNetModel::postProcess(PostProcessContext postContext) {
  qnn::tools::iotensor::IOTensor *iotensor =
      (qnn::tools::iotensor::IOTensor *)postContext.iotensor;
  Qnn_Tensor_t *output = &postContext.outputs[0];
  std::vector<size_t> dims;
  iotensor->fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(output),
                     QNN_TENSOR_GET_RANK(output));
  float *floatBuffer = nullptr;
  iotensor->convertToFloat(&floatBuffer, output);
  model_utils::draw_pose(floatBuffer, this->image, dims);
  model_utils::save_image(this->image, postContext, getModelName());
  return true;
}

std::string HRNetModel::getModelName() { return "hrnet"; }