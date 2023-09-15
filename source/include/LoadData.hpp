#pragma once

#include <fstream>
#include <sstream>

#include "global.hpp"

/* NOTE: Row Major storage for matrices probably makes more sense since the files are row major */
ReturnStatus read_mnist_labels(const std::string&, Eigen::MatrixXd&);
ReturnStatus read_mnist_images(const std::string&, Eigen::MatrixXd&);
ReturnStatus load_parameters(const std::string& filename, Eigen::MatrixXd& matrix);
