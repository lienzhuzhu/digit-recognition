#pragma once

#include <fstream>
#include <sstream>

#include "global.hpp"

ReturnStatus read_mnist_labels(const std::string&, Eigen::MatrixXd&);
ReturnStatus read_mnist_images(const std::string&, Eigen::MatrixXd&);
ReturnStatus load_parameters(const std::string& filename, Eigen::MatrixXd& matrix);
