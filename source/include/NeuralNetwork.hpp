#include "global.hpp"
#include "LoadData.hpp"


Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& z);
void save_parameters(const Eigen::MatrixXd& matrix, const std::string& filename);
ReturnStatus train_nn();
ReturnStatus test_nn(const Eigen::MatrixXd& w_i_h, const Eigen::MatrixXd& b_i_h, const Eigen::MatrixXd& w_h_o, const Eigen::MatrixXd& b_h_o);
int predict(const Eigen::MatrixXd& img, const Eigen::MatrixXd& w_i_h, const Eigen::MatrixXd& b_i_h, const Eigen::MatrixXd& w_h_o, const Eigen::MatrixXd& b_h_o);
