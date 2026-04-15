/*
**************************************************************************************************
* Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause-Clear
**************************************************************************************************
*/

#pragma once
#include <Models/Model.hpp>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <opencv2/core.hpp>

class MidasV2SmallModel : public Model
{
public:
  void preProcess(PreProcessContext preContext);
  bool postProcess(PostProcessContext postContext);
  static std::string getModelName();

private:
	int org_height;
	int org_width;
};
