/*
**************************************************************************************************
* Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause-Clear
**************************************************************************************************
*/
#pragma once

#include <Models/Model.hpp>

template <typename T> Model *createT() { return new T; }

class ModelFactory {

public:
  ModelFactory::ModelFactory();

  typedef std::map<std::string, Model *(*)()> modelClassByName_t;

  bool hasModel(std::string const &s);

  Model *createInstance(std::string const &s);

  modelClassByName_t *getMap();

private:
  modelClassByName_t *map;
};