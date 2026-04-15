/*
**************************************************************************************************
* Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause-Clear
**************************************************************************************************
*/
#pragma once

#include "QnnTypeMacros.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <regex>
#include <unordered_map>
#include <vector>

using namespace std;

typedef struct PreProcessContext {
  uint8_t *buffer;
  vector<string> inputFiles;
  size_t fileIndex;
  size_t inputNodeIndex;
  std::string inputNodeName;
  vector<size_t> dims;

  PreProcessContext(uint8_t *buffer, vector<string> inputFiles,
                    size_t fileIndex, size_t inputNodeIndex,
                    std::string inputNodeName, const vector<size_t>& dims)
      : buffer(buffer), inputFiles(inputFiles), fileIndex(fileIndex),
        inputNodeIndex(inputNodeIndex), inputNodeName(inputNodeName),
        dims(dims) {
    return;
  }
} PreProcessContext;

typedef struct PostProcessContext {
  void *iotensor;
  uint32_t graphIdx;
  size_t startIdx;
  char *graphName;
  Qnn_Tensor_t *outputs;
  uint32_t numOutputs;
  uint32_t graphsCount;
  std::string outputPath;
  size_t numInputFilesPopulated;
  size_t outputBatchSize;

  PostProcessContext(void *iotensor, uint32_t graphIdx, size_t startIdx,
                     char *graphName, Qnn_Tensor_t *outputs,
                     uint32_t numOutputs, uint32_t graphsCount,
                     std::string outputPath, size_t numInputFilesPopulated,
                     size_t outputBatchSize)
      : iotensor(iotensor), graphIdx(graphIdx), startIdx(startIdx),
        graphName(graphName), outputs(outputs), numOutputs(numOutputs),
        graphsCount(graphsCount), outputPath(outputPath),
        numInputFilesPopulated(numInputFilesPopulated),
        outputBatchSize(outputBatchSize) {
    return;
  }
} PostProcessContext;

class Model {
public:
  virtual void preProcess(PreProcessContext preContext) = 0;
  virtual bool postProcess(PostProcessContext postContext) = 0;

  virtual std::unordered_map<string, uint32_t> getInputNameToIndexMap() {
    std::vector<string> input_tensors = getInputTensorNames();
    std::unordered_map<string, uint32_t> inputNameToIndexMap = {};
    int count = 0;
    for (const string& tensorName : input_tensors) {
      inputNameToIndexMap[tensorName] = count;
      inputNameToIndexMap[sanitizeTensorName(tensorName)] = count;
      count++;
    }
    return inputNameToIndexMap;
  }

  virtual std::vector<string> getInputTensorNames() {
    return std::vector<string>{};
  }

  auto getAverageTime(const std::vector<chrono::steady_clock::time_point>& start,
                      const std::vector<chrono::steady_clock::time_point>& stop) {
    auto times = 0;
    int i = 0;
    while (i < start.size()) {
      times +=
          (int)chrono::duration_cast<chrono::milliseconds>(stop.at(i) - start.at(i))
              .count();
      i++;
    }
    return times / start.size();
  }

  virtual void displayKPI() {
    auto avg_pre_duration =
        getAverageTime(preprocess_start_times, preprocess_stop_times);
    auto avg_inference_duration =
        getAverageTime(preprocess_stop_times, postprocess_start_times);
    auto avg_post_duration =
        getAverageTime(postprocess_start_times, postprocess_stop_times);
    auto avg_exe_duration =
        getAverageTime(execution_start_times, execution_stop_times);

    auto model_load_duration = (int)chrono::duration_cast<chrono::milliseconds>(
                                   model_load_stop_time - model_load_start_time)
                                   .count();
    auto sum_of_all_exes_duration =
        (int)chrono::duration_cast<chrono::milliseconds>(
            sum_of_all_executions_stop_time - sum_of_all_executions_start_time)
            .count();

    cout << "Model Load time (ms): " << model_load_duration << endl;
    cout << "Total number of Executions: " << execution_start_times.size()
         << endl;
    cout << "Avg Model Preprocessing time (ms): " << avg_pre_duration << endl;
    cout << "Avg Model Inference time (ms): " << avg_inference_duration << endl;
    cout << "Avg Model Postprocessing time (ms): " << avg_post_duration << endl;
    cout << "Avg Execution time (ms): " << avg_exe_duration << endl;
    cout << "Sum of all Execution times (ms): " << sum_of_all_exes_duration
         << endl;
    cout << "Aprox. FPS: " << round((1.0 / avg_exe_duration) * 1000.0) << endl;
  }

  chrono::steady_clock::time_point model_load_start_time;
  chrono::steady_clock::time_point model_load_stop_time;
  chrono::steady_clock::time_point sum_of_all_executions_start_time;
  chrono::steady_clock::time_point sum_of_all_executions_stop_time;
  vector<chrono::steady_clock::time_point> execution_start_times;
  vector<chrono::steady_clock::time_point> execution_stop_times;
  vector<chrono::steady_clock::time_point> preprocess_start_times;
  vector<chrono::steady_clock::time_point> preprocess_stop_times;
  vector<chrono::steady_clock::time_point> postprocess_start_times;
  vector<chrono::steady_clock::time_point> postprocess_stop_times;

private:
  static std::string sanitizeTensorName(std::string name) {
    std::string sanitizedName =
        std::regex_replace(name, std::regex("\\W+"), "_");
    if (!std::isalpha(sanitizedName[0]) && sanitizedName[0] != '_') {
      sanitizedName = "_" + sanitizedName;
    }
    return sanitizedName;
  }
};
