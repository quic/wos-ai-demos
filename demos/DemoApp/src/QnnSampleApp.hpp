//==============================================================================
//
//  Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
//  SPDX-License-Identifier: BSD-3-Clause-Clear
//
//==============================================================================
#pragma once

#include <memory>
#include <queue>

#include "IOTensor.hpp"
#include "Models/Model.hpp"
#include "SampleApp.hpp"

namespace qnn {
namespace tools {
namespace sample_app {

enum class StatusCode {
  SUCCESS,
  FAILURE,
  FAILURE_INPUT_LIST_EXHAUSTED,
  FAILURE_SYSTEM_ERROR,
  FAILURE_SYSTEM_COMMUNICATION_ERROR,
  QNN_FEATURE_UNSUPPORTED
};

class QnnSampleApp {
public:
  QnnSampleApp(
      Model *model, QnnFunctionPointers qnnFunctionPointers,
      vector<std::string> inputArgs, std::string opPackagePaths,
      void *backendLibraryHandle, std::string outputPath = s_defaultOutputPath,
      bool debug = false,
      iotensor::OutputDataType outputDataType =
          iotensor::OutputDataType::FLOAT_ONLY,
      iotensor::InputDataType inputDataType = iotensor::InputDataType::FLOAT,
      ProfilingLevel profilingLevel = ProfilingLevel::OFF,
      bool dumpOutputs = false, std::string cachedBinaryPath = "",
      std::string saveBinaryName = "", bool needKpi = false);

  // @brief Print a message to STDERR then return a nonzero
  //  exit status.
  static int32_t reportError(const std::string &err);

  StatusCode initialize();

  StatusCode initializeBackend();

  StatusCode createContext();

  StatusCode composeGraphs();

  StatusCode finalizeGraphs();

  StatusCode executeGraphs();

  StatusCode registerOpPackages();

  StatusCode createFromBinary();

  StatusCode saveBinary();

  StatusCode freeContext();

  StatusCode terminateBackend();

  StatusCode freeGraphs();

  Qnn_ContextHandle_t getContext();

  StatusCode initializeProfiling();

  std::string getBackendBuildId();

  StatusCode isDevicePropertySupported();

  StatusCode createPowerConfigId();

  StatusCode setPowerConfig();

  StatusCode destroyPowerConfigId();

  StatusCode createDevice();

  StatusCode freeDevice();

  void displayKPI();

  static StatusCode verifyFailReturnStatus(Qnn_ErrorHandle_t errCode);

  virtual ~QnnSampleApp();

  Model *m_model;

private:
  StatusCode extractBackendProfilingInfo(Qnn_ProfileHandle_t profileHandle);

  StatusCode extractProfilingSubEvents(QnnProfile_EventId_t profileEventId);

  StatusCode extractProfilingEvent(QnnProfile_EventId_t profileEventId);

  static const std::string s_defaultOutputPath;

  QnnFunctionPointers m_qnnFunctionPointers;
  vector<std::string> m_inputArgs;
  std::vector<std::vector<std::vector<std::string>>> m_inputimgLists;
  std::vector<std::unordered_map<std::string, uint32_t>> m_inputNameToIndex;
  std::vector<std::string> m_opPackagePaths;
  std::string m_outputPath;
  std::string m_saveBinaryName;
  std::string m_cachedBinaryPath;
  QnnBackend_Config_t **m_backendConfig = nullptr;
  Qnn_ContextHandle_t m_context = nullptr;
  QnnContext_Config_t **m_contextConfig = nullptr;
  bool m_debug;
  iotensor::OutputDataType m_outputDataType;
  iotensor::InputDataType m_inputDataType;
  ProfilingLevel m_profilingLevel;
  bool m_dumpOutputs;
  qnn_wrapper_api::GraphInfo_t **m_graphsInfo;
  uint32_t m_graphsCount;
  void *m_backendLibraryHandle;
  iotensor::IOTensor m_ioTensor;
  bool m_isBackendInitialized;
  bool m_isContextCreated;
  Qnn_ProfileHandle_t m_profileBackendHandle = nullptr;
  qnn_wrapper_api::GraphConfigInfo_t **m_graphConfigsInfo = nullptr;
  uint32_t m_graphConfigsInfoCount;
  Qnn_LogHandle_t m_logHandle = nullptr;
  Qnn_BackendHandle_t m_backendHandle = nullptr;
  Qnn_DeviceHandle_t m_deviceHandle = nullptr;
  bool m_needKpi;

  uint32_t powerConfigId;
  uint32_t deviceId = 0;
  uint32_t coreId = 0;
};
} // namespace sample_app
} // namespace tools
} // namespace qnn
