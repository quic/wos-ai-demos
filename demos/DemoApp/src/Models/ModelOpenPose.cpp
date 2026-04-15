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
#include <Models/ModelOpenPose.hpp>
#include <Models/Model.hpp>
#include <Models/ModelUtils.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>


void OpenPoseModel::preProcess(PreProcessContext preContext) {

    const int width = (int)preContext.dims[2];
    const int height = (int)preContext.dims[1];
    const int channel = (int)preContext.dims[3];
    cv::Mat resized_img;
    cv::Mat converted_img;
    image =
        cv::imread(preContext.inputFiles[preContext.fileIndex], cv::IMREAD_COLOR);
    cv::resize(this->image, resized_img, cv::Size((int)width, (int)height));
    resized_img.convertTo(converted_img, CV_32FC2, 1.0 / 255.0);
    std::memcpy(preContext.buffer, reinterpret_cast<uint8_t*>(converted_img.data),
        width * channel * height * sizeof(float));
}

bool OpenPoseModel::postProcess(PostProcessContext postContext) {
    qnn::tools::iotensor::IOTensor* iotensor =
        (qnn::tools::iotensor::IOTensor*)postContext.iotensor;
    // two output - Heatmap and PAF data
    Qnn_Tensor_t *heatmap = &postContext.outputs[1];
    Qnn_Tensor_t *paf = &postContext.outputs[0];
    // to fetch the input image dimension
    std::vector<size_t> heat_dims;
    std::vector<size_t> paf_dims;
    iotensor->fillDims(heat_dims, QNN_TENSOR_GET_DIMENSIONS(heatmap),
        QNN_TENSOR_GET_RANK(heatmap));
    iotensor->fillDims(paf_dims, QNN_TENSOR_GET_DIMENSIONS(paf),
        QNN_TENSOR_GET_RANK(paf));

    float* heat_Buffer = reinterpret_cast<float*>(QNN_TENSOR_GET_CLIENT_BUF(heatmap).data);
    float* paf_Buffer = reinterpret_cast<float*>(QNN_TENSOR_GET_CLIENT_BUF(paf).data);
    model_utils::extractPoses(heat_Buffer, paf_Buffer, this->image, heat_dims, paf_dims);
    model_utils::save_image(this->image, postContext, getModelName());
    return true;
}

std::string OpenPoseModel::getModelName() { return "openpose"; }