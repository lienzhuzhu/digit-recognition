#pragma once

#include "global.hpp"


Eigen::MatrixXd read_mnist_labels(const std::string& full_path);
Eigen::MatrixXd read_mnist_images(const std::string& full_path);
//ReturnStatus load_data(Eigen::MatrixXd&, Eigen::MatrixXd&);
