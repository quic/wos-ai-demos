//==============================================================================
//
//  Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
//  SPDX-License-Identifier: BSD-3-Clause-Clear
//
//==============================================================================

#pragma once

#include "SampleApp.hpp"

namespace qnn {
namespace tools {
namespace dynamicloadutil {
enum class StatusCode {
  SUCCESS,
  FAILURE,
  FAIL_LOAD_BACKEND,
  FAIL_LOAD_MODEL,
  FAIL_SYM_FUNCTION,
  FAIL_GET_INTERFACE_PROVIDERS,
  FAIL_LOAD_SYSTEM_LIB,
};

StatusCode
getQnnFunctionPointers(const std::string& backendPath, const std::string& modelPath,
                       sample_app::QnnFunctionPointers *qnnFunctionPointers,
                       void **backendHandleRtn, bool loadModelLib,
                       void **modelHandleRtn);
StatusCode getQnnSystemFunctionPointers(
    const std::string& systemLibraryPath,
    sample_app::QnnFunctionPointers *qnnFunctionPointers);
} // namespace dynamicloadutil
} // namespace tools
} // namespace qnn
