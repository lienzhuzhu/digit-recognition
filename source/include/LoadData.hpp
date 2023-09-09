#pragma once

#include "global.hpp"

std::vector<uint8_t> read_mnist_labels(const std::string& full_path);
std::vector<std::vector<uint8_t>> read_mnist_images(const std::string& full_path);
