#==============================================================================
#
#  Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
#  SPDX-License-Identifier: BSD-3-Clause-Clear
#
#==============================================================================

set(APP "qnn-demo-app")
set(APP_SOURCES "QnnSampleApp.cpp"
                "main.cpp"
                "Log/Logger.cpp"
                "Log/LogUtils.cpp"
                "PAL/src/windows/Common.cpp"
                "PAL/src/windows/Directory.cpp"
                "PAL/src/windows/DynamicLoading.cpp"
                "PAL/src/windows/FileOp.cpp"
                "PAL/src/windows/Path.cpp"
                "PAL/src/common/GetOpt.cpp"
                "PAL/src/common/StringOp.cpp"
                "Utils/DataUtil.cpp"
                "Utils/DynamicLoadUtil.cpp"
                "Utils/IOTensor.cpp"
                "Utils/QnnSampleAppUtils.cpp"
                "WrapperUtils/QnnWrapperUtils.cpp"
                "Models/DexiNedModel.cpp"
                "Models/DMSHNModel.cpp"
                "Models/QuickSRNetModel.cpp"
                "Models/MidasV2SmallModel.cpp"
                "Models/DeepLabV3Model.cpp"
                "Models/HRNetModel.cpp"
                "Models/DeepFillV2Model.cpp"
                "Models/HeadPoseModel.cpp"
                "Models/ModelOpenPose.cpp"
                "Models/ModelUtils.cpp"
                "Models/ModelFactory.cpp")

add_executable(${APP} ${APP_SOURCES})

target_compile_definitions(${APP} PUBLIC "-DNOMINMAX")
target_link_libraries(${APP} PRIVATE Shlwapi Shell32)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT /O2 /Ob3")

find_package(OpenCV REQUIRED)
if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )
  # Additional Library Directories
  link_directories( ${OpenCV_LIB_DIR} )
endif()
message(OPENCV_LIBS ${OpenCV_LIBS})

set(QNN_SDK_ROOT $ENV{QNN_SDK_ROOT})
message(QNN_SDK_ROOT="${QNN_SDK_ROOT}")
target_include_directories(${APP} PUBLIC CachingUtil
                                         Log
                                         PAL/include
                                         Utils
                                         WrapperUtils
                                         ${CMAKE_BINARY_DIR}
                                         ${QNN_SDK_ROOT}/include/QNN
                                         ./)
target_link_libraries(${APP} PUBLIC ${OpenCV_LIBS})
