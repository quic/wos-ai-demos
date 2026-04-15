/*
**************************************************************************************************
* Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause-Clear
**************************************************************************************************
*/
#pragma once

#include <Models/Model.hpp>

class DexiNedModel : public Model {
public:
  void preProcess(PreProcessContext preContext);
  bool postProcess(PostProcessContext postContext);
  static std::string getModelName();
};