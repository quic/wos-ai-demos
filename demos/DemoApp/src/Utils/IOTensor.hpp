//==============================================================================
//
//  Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
//  SPDX-License-Identifier: BSD-3-Clause-Clear
//
//==============================================================================
#pragma once

#include <memory>
#include <queue>

#include "Models/Model.hpp"
#include "QnnBackend.h"
#include "QnnCommon.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnProperty.h"
#include "QnnSampleAppUtils.hpp"
#include "QnnTensor.h"
#include "QnnTypes.h"
#include "QnnWrapperUtils.hpp"

namespace qnn {
namespace tools {
namespace iotensor {

enum class StatusCode { SUCCESS, FAILURE };
enum class OutputDataType {
  FLOAT_ONLY,
  NATIVE_ONLY,
  FLOAT_AND_NATIVE,
  INVALID
};
enum class InputDataType { FLOAT, NATIVE, INVALID };

OutputDataType parseOutputDataType(std::string dataTypeString);
InputDataType parseInputDataType(std::string dataTypeString);

using PopulateInputTensorsRetType_t = std::tuple<StatusCode, size_t, size_t>;

class IOTensor {
public:
  static StatusCode setupInputAndOutputTensors(Qnn_Tensor_t **inputs,
                                        Qnn_Tensor_t **outputs,
                                        qnn_wrapper_api::GraphInfo_t graphInfo);

#ifndef __hexagon__
  StatusCode writeOutputTensors(Model *model, uint32_t graphIdx,
                                size_t startIdx, char *graphName,
                                Qnn_Tensor_t *outputs, uint32_t numOutputs,
                                OutputDataType outputDatatype,
                                uint32_t graphsCount, std::string outputPath,
                                size_t numInputFilesPopulated,
                                size_t outputBatchSize);
#endif

  PopulateInputTensorsRetType_t populateInputTensors(
      Model *model, uint32_t graphIdx,
      const std::vector<std::vector<std::string>> &filePathsVector,
      const size_t filePathsIndexOffset, const bool loopBackToStart,
      const std::unordered_map<std::string, uint32_t> &inputNameToIndex,
      Qnn_Tensor_t *inputs, qnn_wrapper_api::GraphInfo_t graphInfo,
      iotensor::InputDataType inputDataType);

  static StatusCode tearDownInputAndOutputTensors(Qnn_Tensor_t **inputs,
                                           Qnn_Tensor_t **outputs,
                                           size_t numInputTensors,
                                           size_t numOutputTensors);

  PopulateInputTensorsRetType_t
  populateInputTensor(Model *model, const std::vector<std::string> &filePaths,
                      const size_t filePathsIndexOffset,
                      const bool loopBackToStart, Qnn_Tensor_t *input,
                      InputDataType inputDataType, size_t inputIndex,
                      const std::string &inputNodeName);

  static PopulateInputTensorsRetType_t readDataAndAllocateBuffer(
      Model *model, const std::vector<std::string> &filePaths,
      const size_t filePathsIndexOffset, const bool loopBackToStart,
      const std::vector<size_t> &dims, Qnn_DataType_t dataType, uint8_t **bufferToCopy,
      size_t inputIndex, const std::string &inputNodeName);

  template <typename T>
  static StatusCode allocateBuffer(T **buffer, size_t &elementCount);

#ifndef __hexagon__
  StatusCode convertToFloat(float **out, Qnn_Tensor_t *tensor);

  StatusCode convertAndWriteOutputTensorInFloat(
      Qnn_Tensor_t *output, const std::vector<std::string> &outputPaths,
      const std::string &fileName, size_t outputBatchSize);

  static StatusCode writeOutputTensor(Qnn_Tensor_t *output,
                               const std::vector<std::string> &outputPaths,
                               const std::string &fileName, size_t outputBatchSize);
#endif

  static StatusCode allocateAndCopyBuffer(uint8_t **buffer, Qnn_Tensor_t *tensor);

  static StatusCode tearDownTensors(Qnn_Tensor_t *tensors, uint32_t tensorCount);

  static StatusCode allocateBuffer(uint8_t **buffer, const std::vector<size_t> &dims,
                            Qnn_DataType_t dataType);

  StatusCode copyFromFloatToNative(float *floatBuffer, Qnn_Tensor_t *tensor);

  static StatusCode setupTensors(Qnn_Tensor_t **tensors, uint32_t tensorCount,
                          const Qnn_Tensor_t *tensorWrappers);

  static StatusCode fillDims(std::vector<size_t> &dims, uint32_t *inDimensions,
                      uint32_t rank);
};
} // namespace iotensor
} // namespace tools
} // namespace qnn