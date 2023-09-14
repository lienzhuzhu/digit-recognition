#include "include/NeuralNetwork.hpp"


#define NUM_INPUT_NEURONS   784
#define NUM_HIDDEN_NEURONS  20
#define NUM_OUTPUT_NEURONS  10


Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& z) {
    return 1.0 / (1.0 + (-z).array().exp());
}


void save_parameters(const Eigen::MatrixXd& matrix, const std::string& filename) {
    std::ofstream out_file(filename);
    if (out_file.is_open()) {
        out_file << matrix.format(Eigen::IOFormat(Eigen::FullPrecision));
        out_file.close();
    } else {
        std::cout << "Cannot open file for writing: " << filename << std::endl;
    }
}

int predict(const Eigen::MatrixXd& img, const Eigen::MatrixXd& w_i_h, const Eigen::MatrixXd& b_i_h, const Eigen::MatrixXd& w_h_o, const Eigen::MatrixXd& b_h_o) {
    Eigen::VectorXd::Index predicted_index, max_col;

    Eigen::MatrixXd h_pre = b_i_h + w_i_h * img; // NOTE: make sure img is a column vector or a <rows> x 1 matrix
    Eigen::MatrixXd h = sigmoid(h_pre);

    Eigen::MatrixXd o_pre = b_h_o + w_h_o * h;
    Eigen::MatrixXd o = sigmoid(o_pre);

    o.maxCoeff(&predicted_index, &max_col);

    return predicted_index;
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

            /* Where the learning happens */
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

    std::cout << "w_i_h rows, cols\t" << w_i_h.rows() << ",\t" << w_i_h.cols() << std::endl;
    std::cout << "b_i_h rows, cols\t" << b_i_h.rows() << ",\t" << b_i_h.cols() << std::endl;
    std::cout << "w_h_o rows, cols\t" << w_h_o.rows() << ",\t" << w_h_o.cols() << std::endl;
    std::cout << "b_h_o rows, cols\t" << b_h_o.rows() << ",\t" << b_h_o.cols() << std::endl;

    save_parameters(w_i_h, "model/w_i_h.txt");
    save_parameters(b_i_h, "model/b_i_h.txt");
    save_parameters(w_h_o, "model/w_h_o.txt");
    save_parameters(b_h_o, "model/b_h_o.txt");

    return SUCCESS;
}


ReturnStatus test_nn(const Eigen::MatrixXd& w_i_h, const Eigen::MatrixXd& b_i_h, const Eigen::MatrixXd& w_h_o, const Eigen::MatrixXd& b_h_o) {
    Eigen::MatrixXd images, labels;
    Eigen::VectorXd::Index maxRow, maxCol;
    int prediction, true_index;
    
    if (read_mnist_labels(TEST_LABELS_PATH, labels)) {
        std::cout << "Reading labels failed" << std::endl;
        return FAILURE;
    }

    if (read_mnist_images(TEST_IMAGES_PATH, images)) {
        std::cout << "Reading images failed" << std::endl;
        return FAILURE;
    }

    int numCorrect = 0;

    for (int i = 0; i < images.rows(); ++i) { // Assuming each row is a different image
        Eigen::MatrixXd img = images.row(i).transpose();
        Eigen::MatrixXd label = labels.row(i).transpose();

        prediction = predict(img, w_i_h, b_i_h, w_h_o, b_h_o);

        label.maxCoeff(&maxRow, &maxCol);
        true_index = maxRow;
        
        numCorrect += (prediction == true_index);
    }

    double accuracy = static_cast<double>(numCorrect) / images.rows() * 100.0;
    std::cout << "Model accuracy on test set: " << accuracy << "%" << std::endl;

    return SUCCESS;
}
