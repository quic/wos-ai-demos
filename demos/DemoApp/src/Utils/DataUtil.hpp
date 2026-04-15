//==============================================================================
//
//  Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
//  SPDX-License-Identifier: BSD-3-Clause-Clear
//
//==============================================================================
#pragma once

#include <map>
#include <queue>
#include <vector>

#include "QnnTypes.h"
#include <Models/Model.hpp>

namespace qnn {
namespace tools {
namespace datautil {
enum class StatusCode {
  SUCCESS,
  DATA_READ_FAIL,
  DATA_WRITE_FAIL,
  FILE_OPEN_FAIL,
  DIRECTORY_CREATE_FAIL,
  INVALID_DIMENSIONS,
  INVALID_DATA_TYPE,
  DATA_SIZE_MISMATCH,
  INVALID_BUFFER,
};

const size_t g_bitsPerByte = 8;

using ReadBatchDataRetType_t = std::tuple<StatusCode, size_t, size_t>;

std::tuple<StatusCode, size_t> getDataTypeSizeInBytes(Qnn_DataType_t dataType);

std::tuple<StatusCode, size_t> calculateLength(const std::vector<size_t>& dims,
                                               Qnn_DataType_t dataType);

size_t calculateElementCount(std::vector<size_t> dims);

std::tuple<StatusCode, size_t> getFileSize(std::string filePath);

StatusCode readDataFromFile(std::string filePath, const std::vector<size_t>& dims,
                            Qnn_DataType_t dataType, uint8_t *buffer);

/*
 * Read data in batches from vector and try to matches the model input's
 * batches. If the vector is empty while matching the batch size of model,
 * pad the remaining buffer with zeros
 * @param filePaths image paths vector
 * @param fileIndex index offset in the vector
 * @param loopBackToStart loop the vector to fill the remaining tensor data
 * @param dims model input dimensions
 * @param dataType to create input buffer from file
 * @param buffer to fill the input image data
 *
 * @return ReadBatchDataRetType_t returns numFilesCopied and batchSize along
 * with status
 */
ReadBatchDataRetType_t
readBatchData(Model *model, const std::vector<std::string> &filePaths,
              const size_t fileIndex, const bool loopBackToStart,
              const std::vector<size_t> &dims, const Qnn_DataType_t dataType,
              uint8_t *buffer, size_t inputIndex, std::string inputNodeName);

StatusCode readBinaryFromFile(std::string filePath, uint8_t *buffer,
                              size_t bufferSize);

#ifndef __hexagon__
StatusCode writeDataToFile(std::string fileDir, const std::string& fileName,
                           const std::vector<size_t>& dims, Qnn_DataType_t dataType,
                           const uint8_t *buffer);

StatusCode writeBatchDataToFile(const std::vector<std::string>& fileDirs,
                                const std::string& fileName, const std::vector<size_t>& dims,
                                Qnn_DataType_t dataType, const uint8_t *buffer,
                                const size_t batchSize);

StatusCode writeBinaryToFile(std::string fileDir, const std::string& fileName,
                             const uint8_t *buffer, size_t bufferSize);
#endif

template <typename T_QuantType>
datautil::StatusCode floatToTfN(T_QuantType *out, float *in, int32_t offset,
                                float scale, size_t numElements);

template <typename T_QuantType>
datautil::StatusCode tfNToFloat(float *out, T_QuantType *in, int32_t offset,
                                float scale, size_t numElements);

template <typename T_QuantType>
datautil::StatusCode castToFloat(float *out, T_QuantType *in,
                                 size_t numElements);

template <typename T_QuantType>
datautil::StatusCode castFromFloat(T_QuantType *out, float *in,
                                   size_t numElements);

const std::map<Qnn_DataType_t, size_t> g_dataTypeToSize = {
    {QNN_DATATYPE_INT_8, 1},           {QNN_DATATYPE_INT_16, 2},
    {QNN_DATATYPE_INT_32, 4},          {QNN_DATATYPE_INT_64, 8},
    {QNN_DATATYPE_UINT_8, 1},          {QNN_DATATYPE_UINT_16, 2},
    {QNN_DATATYPE_UINT_32, 4},         {QNN_DATATYPE_UINT_64, 8},
    {QNN_DATATYPE_FLOAT_16, 2},        {QNN_DATATYPE_FLOAT_32, 4},
    {QNN_DATATYPE_FLOAT_64, 8},        {QNN_DATATYPE_SFIXED_POINT_8, 1},
    {QNN_DATATYPE_SFIXED_POINT_16, 2}, {QNN_DATATYPE_SFIXED_POINT_32, 4},
    {QNN_DATATYPE_UFIXED_POINT_8, 1},  {QNN_DATATYPE_UFIXED_POINT_16, 2},
    {QNN_DATATYPE_UFIXED_POINT_32, 4}, {QNN_DATATYPE_BOOL_8, 1},
};
} // namespace datautil
} // namespace tools
} // namespace qnn
