//==============================================================================
//
//  Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
//  SPDX-License-Identifier: BSD-3-Clause-Clear
//
//==============================================================================

#pragma once

#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>

#include "QnnLog.h"

namespace qnn {
namespace log {
namespace utils {

void logStdoutCallback(const char *fmt, QnnLog_Level_t level,
                       uint64_t timestamp, va_list argp);
static std::mutex sg_logUtilMutex;

} // namespace utils
} // namespace log
} // namespace qnn
