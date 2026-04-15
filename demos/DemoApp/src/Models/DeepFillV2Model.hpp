/*
**************************************************************************************************
* Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause-Clear
**************************************************************************************************
*/
#pragma once

#include <Models/Model.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

class DeepFillV2Model : public Model {
public:
  void preProcess(PreProcessContext preContext);
  bool postProcess(PostProcessContext postContext);
  static std::string getModelName();
  std::vector<std::string> getInputTensorNames();

private:
  const std::string INPUT_0_NAME = "onnx::Mul_0";
  const std::string INPUT_1_NAME = "mask";

  cv::Mat input0, input1;
};