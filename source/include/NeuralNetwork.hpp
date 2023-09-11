#include <Eigen/Dense>
#include "global.hpp"


Eigen::MatrixXd sigmoid(Eigen::MatrixXd z) {
    return 1.0 / (1.0 + (-z).array().exp());
}
