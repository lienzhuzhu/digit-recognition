#include "include/NeuralNetwork.hpp"
#include "include/LoadData.hpp"


#define NUM_INPUT_NEURONS   784
#define NUM_HIDDEN_NEURONS  20
#define NUM_OUTPUT_NEURONS  10


Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& z) {
    return 1.0 / (1.0 + (-z).array().exp());
}


void saveMatrix(const Eigen::MatrixXd& matrix, const std::string& filename) {
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        outFile << matrix.format(Eigen::IOFormat(Eigen::FullPrecision));
        outFile.close();
    } else {
        std::cout << "Cannot open file for writing: " << filename << std::endl;
    }
}


ReturnStatus train_nn() {
    Eigen::MatrixXd images, labels;
    
    if (read_mnist_labels(TRAIN_LABELS_PATH, labels)) {
        std::cout << "reading labels failed" << std::endl;
        return FAILURE;
    }
    if (read_mnist_images(TRAIN_IMAGES_PATH, images)) {
        std::cout << "reading images failed" << std::endl;
        return FAILURE;
    }

    // Randomly initialize weights and biases
    Eigen::MatrixXd w_i_h = Eigen::MatrixXd::Random(NUM_HIDDEN_NEURONS, NUM_INPUT_NEURONS);
    Eigen::MatrixXd b_i_h = Eigen::MatrixXd::Zero(NUM_HIDDEN_NEURONS, 1);

    Eigen::MatrixXd w_h_o = Eigen::MatrixXd::Random(NUM_OUTPUT_NEURONS, NUM_HIDDEN_NEURONS);
    Eigen::MatrixXd b_h_o = Eigen::MatrixXd::Zero(NUM_OUTPUT_NEURONS, 1);

    double learn_rate = 0.01;
    int nr_correct = 0;
    int epochs = 3;

    int predicted_index, true_index;
    Eigen::VectorXd::Index maxRow, maxCol;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "Training Epoch: " << epoch << std::endl;
        for (int i = 0; i < images.rows(); ++i) {
            Eigen::MatrixXd img = images.row(i).transpose();
            Eigen::MatrixXd label = labels.row(i).transpose();

            // Forward propagation input -> hidden
            Eigen::MatrixXd h_pre = b_i_h + w_i_h * img;
            Eigen::MatrixXd h = sigmoid(h_pre);

            // Forward propagation hidden -> output
            Eigen::MatrixXd o_pre = b_h_o + w_h_o * h;
            Eigen::MatrixXd o = sigmoid(o_pre);

            // Find the index of the maximum value in the output layer's activation
            o.maxCoeff(&maxRow, &maxCol);
            predicted_index = maxRow;

            // Find the index of the maximum value in the label (should be 1)
            label.maxCoeff(&maxRow, &maxCol);
            true_index = maxRow;

            // Count it as correct if the indices match
            nr_correct += (predicted_index == true_index);

            // Backpropagation output -> hidden
            Eigen::MatrixXd delta_o = o - label;
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

    // Save learned parameters
    saveMatrix(w_i_h, "model/w_i_h.txt");
    saveMatrix(b_i_h, "model/b_i_h.txt");
    saveMatrix(w_h_o, "model/w_h_o.txt");
    saveMatrix(b_h_o, "model/b_h_o.txt");

    return SUCCESS;
}


void test_nn() {
}
