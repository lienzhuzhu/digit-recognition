#include "global.hpp"


Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& z);
void saveMatrix(const Eigen::MatrixXd& matrix, const std::string& filename);
ReturnStatus train_nn();
void test_nn();
