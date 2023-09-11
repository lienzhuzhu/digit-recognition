#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

// Sigmoid function
Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& z) {
    return 1.0 / (1.0 + (-z).array().exp());
}

void train_nn() {
    // Your get_mnist function here. Assuming images and labels are Eigen::MatrixXd
    Eigen::MatrixXd images, labels;
    // get_mnist(images, labels);

    Eigen::MatrixXd w_i_h = Eigen::MatrixXd::Random(20, 784);
    Eigen::MatrixXd w_h_o = Eigen::MatrixXd::Random(10, 20);
    Eigen::MatrixXd b_i_h = Eigen::MatrixXd::Zero(20, 1);
    Eigen::MatrixXd b_h_o = Eigen::MatrixXd::Zero(10, 1);

    double learn_rate = 0.01;
    int nr_correct = 0;
    int epochs = 3;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Assuming one image and label at a time
        for (int i = 0; i < images.rows(); ++i) {
            Eigen::MatrixXd img = images.row(i).transpose();
            Eigen::MatrixXd l = labels.row(i).transpose();

            // Forward propagation input -> hidden
            Eigen::MatrixXd h_pre = b_i_h + w_i_h * img;
            Eigen::MatrixXd h = sigmoid(h_pre);

            // Forward propagation hidden -> output
            Eigen::MatrixXd o_pre = b_h_o + w_h_o * h;
            Eigen::MatrixXd o = sigmoid(o_pre);

            // Error calculation (mean squared error)
            Eigen::MatrixXd e = (o - l).array().pow(2).sum() / o.size();

            nr_correct += (o.maxCoeff() == l.maxCoeff());

            // Backpropagation output -> hidden
            Eigen::MatrixXd delta_o = o - l;
            w_h_o += -learn_rate * delta_o * h.transpose();
            b_h_o += -learn_rate * delta_o;

            // Backpropagation hidden -> input
            Eigen::MatrixXd delta_h = w_h_o.transpose() * delta_o;
            delta_h = delta_h.array() * (h.array() * (1 - h.array()));
            w_i_h += -learn_rate * delta_h * img.transpose();
            b_i_h += -learn_rate * delta_h;
        }

        std::cout << "Epoch: " << epoch << " Accuracy: " << (double)nr_correct / images.rows() * 100 << "%" << std::endl;
        nr_correct = 0;
    }

    // Testing and displaying the results would go here
    
    // Initialize variables for testing
    Eigen::MatrixXd test_images, test_labels;
    // Assuming get_mnist_test is a function to load test images and labels
    // get_mnist(test_images, test_labels);

    int test_nr_correct = 0;

    // Loop through the test set
    for (int i = 0; i < test_images.rows(); ++i) {
        Eigen::MatrixXd test_img = test_images.row(i).transpose();
        Eigen::MatrixXd test_l = test_labels.row(i).transpose();

        // Forward propagation input -> hidden
        Eigen::MatrixXd test_h_pre = b_i_h + w_i_h * test_img;
        Eigen::MatrixXd test_h = sigmoid(test_h_pre);

        // Forward propagation hidden -> output
        Eigen::MatrixXd test_o_pre = b_h_o + w_h_o * test_h;
        Eigen::MatrixXd test_o = sigmoid(test_o_pre);

        // Count correct predictions
        test_nr_correct += (test_o.maxCoeff() == test_l.maxCoeff());
    }

    // Calculate and display the test accuracy
    std::cout << "Test Accuracy: " << (double)test_nr_correct / test_images.rows() * 100 << "%" << std::endl;

}

