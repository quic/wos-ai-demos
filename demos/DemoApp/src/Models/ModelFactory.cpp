/*
**************************************************************************************************
* Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause-Clear
**************************************************************************************************
*/

#include <Models/ModelFactory.hpp>

ModelFactory::ModelFactory() { map = NULL; }

bool ModelFactory::hasModel(std::string const &s) {
  modelClassByName_t::iterator it = getMap()->find(s);
  if (it == getMap()->end())
    return false;
  return true;
}

Model *ModelFactory::createInstance(std::string const &s) {
  modelClassByName_t::iterator it = getMap()->find(s);
  if (it == getMap()->end())
    return 0;
  return it->second();
}

ModelFactory::modelClassByName_t *ModelFactory::getMap() {
  if (!map) {
    map = new modelClassByName_t;
  }
  return map;
}