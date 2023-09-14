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

int predict(const Eigen::MatrixXd& img, const Eigen::MatrixXd& hidden_weights, const Eigen::MatrixXd& hidden_biases, const Eigen::MatrixXd& output_weights, const Eigen::MatrixXd& output_biases) {
    Eigen::VectorXd::Index predicted_index, max_col;

    Eigen::MatrixXd h_pre = hidden_biases + hidden_weights * img; // NOTE: make sure img is a column vector or a <rows> x 1 matrix
    Eigen::MatrixXd h = sigmoid(h_pre);

    Eigen::MatrixXd o_pre = output_biases + output_weights * h;
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
    Eigen::MatrixXd hidden_weights = Eigen::MatrixXd::Random(NUM_HIDDEN_NEURONS, NUM_INPUT_NEURONS);
    Eigen::MatrixXd hidden_biases = Eigen::MatrixXd::Zero(NUM_HIDDEN_NEURONS, 1);

    Eigen::MatrixXd output_weights = Eigen::MatrixXd::Random(NUM_OUTPUT_NEURONS, NUM_HIDDEN_NEURONS);
    Eigen::MatrixXd output_biases = Eigen::MatrixXd::Zero(NUM_OUTPUT_NEURONS, 1);

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
            Eigen::MatrixXd h_pre = hidden_biases + hidden_weights * img;
            Eigen::MatrixXd h = sigmoid(h_pre);

            // Forward propagation hidden -> output
            Eigen::MatrixXd o_pre = output_biases + output_weights * h;
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
            output_weights += -learn_rate * delta_o * h.transpose();
            output_biases += -learn_rate * delta_o;

            // Backpropagation hidden -> input
            Eigen::MatrixXd delta_h = output_weights.transpose() * delta_o;
            delta_h = delta_h.array() * (h.array() * (1 - h.array()));
            hidden_weights += -learn_rate * delta_h * img.transpose();
            hidden_biases += -learn_rate * delta_h;
        }

        std::cout << "Epoch: " << epoch << " Accuracy: " << (double)nr_correct / images.rows() * 100 << "%" << std::endl;
        nr_correct = 0;
    }

    std::cout << "hidden_weights rows, cols\t" << hidden_weights.rows() << ",\t" << hidden_weights.cols() << std::endl;
    std::cout << "hidden_biases rows, cols\t" << hidden_biases.rows() << ",\t" << hidden_biases.cols() << std::endl;
    std::cout << "output_weights rows, cols\t" << output_weights.rows() << ",\t" << output_weights.cols() << std::endl;
    std::cout << "output_biases rows, cols\t" << output_biases.rows() << ",\t" << output_biases.cols() << std::endl;

    save_parameters(hidden_weights, "model/hidden_weights.txt");
    save_parameters(hidden_biases, "model/hidden_biases.txt");
    save_parameters(output_weights, "model/output_weights.txt");
    save_parameters(output_biases, "model/output_biases.txt");

    return SUCCESS;
}


ReturnStatus test_nn(const Eigen::MatrixXd& hidden_weights, const Eigen::MatrixXd& hidden_biases, const Eigen::MatrixXd& output_weights, const Eigen::MatrixXd& output_biases) {
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

        prediction = predict(img, hidden_weights, hidden_biases, output_weights, output_biases);

        label.maxCoeff(&maxRow, &maxCol);
        true_index = maxRow;
        
        numCorrect += (prediction == true_index);
    }

    double accuracy = static_cast<double>(numCorrect) / images.rows() * 100.0;
    std::cout << "Model accuracy on test set: " << accuracy << "%" << std::endl;

    return SUCCESS;
}
