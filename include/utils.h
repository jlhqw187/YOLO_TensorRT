#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "config.h"

static inline bool isFileExists(const std::string &file_name) {
  std::ifstream fin(file_name);
  if (fin) {
    return true;
  } else {
    std::cout << "The file is not exist: " << file_name << std::endl;
    return false;
  }
  return true;
}

