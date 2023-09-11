#include "global.hpp"
#include "LoadData.hpp"


Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& z);
void save_parameters(const Eigen::MatrixXd& matrix, const std::string& filename);
ReturnStatus train_nn();
void test_nn();
