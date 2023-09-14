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


int predict(const Eigen::VectorXd& img, const Eigen::MatrixXd& hidden_weights, const Eigen::MatrixXd& hidden_biases, const Eigen::MatrixXd& output_weights, const Eigen::MatrixXd& output_biases) {
    Eigen::VectorXd::Index predicted_index;

    Eigen::VectorXd hidden_preactivation = hidden_biases + hidden_weights * img;
    Eigen::VectorXd hidden_activation = sigmoid(hidden_preactivation);

    Eigen::VectorXd output_preactivation = output_biases + output_weights * hidden_activation;
    Eigen::VectorXd output_activation = sigmoid(output_preactivation);

    output_activation.maxCoeff(&predicted_index);

    return predicted_index;
}


ReturnStatus test_nn(const Eigen::MatrixXd& hidden_weights, const Eigen::MatrixXd& hidden_biases, const Eigen::MatrixXd& output_weights, const Eigen::MatrixXd& output_biases) {
    Eigen::MatrixXd images, labels;
    
    if (read_mnist_labels(TEST_LABELS_PATH, labels))
        ERROR("Reading Test Labels");

    if (read_mnist_images(TEST_IMAGES_PATH, images))
        ERROR("Reading Test Images");

    int prediction, true_index;
    int num_correct = 0;

    Eigen::MatrixXd hidden_preactivation = hidden_weights * images.transpose() + hidden_biases * Eigen::MatrixXd::Ones(1, images.transpose().cols());
    Eigen::MatrixXd hidden_activation = sigmoid(hidden_preactivation);

    Eigen::MatrixXd output_preactivation = output_weights * hidden_activation + output_biases * Eigen::MatrixXd::Ones(1, hidden_activation.cols());
    Eigen::MatrixXd output_activation = sigmoid(output_preactivation);

    for (int i = 0; i < labels.rows(); ++i) {
        Eigen::VectorXd output_vector = output_activation.col(i);
        Eigen::VectorXd label = labels.row(i).transpose();

        output_vector.maxCoeff(&prediction);
        label.maxCoeff(&true_index);
        
        num_correct += (prediction == true_index);
    }

    
    std::cout << "Model accuracy on test set: " << static_cast<double>(num_correct) / images.rows() * 100.0 << "%" << std::endl;

    return SUCCESS;
}


ReturnStatus train_nn() {
    Eigen::MatrixXd images, labels;
    
    if (read_mnist_labels(TRAIN_LABELS_PATH, labels))
        ERROR("Reading Training Labels");
    if (read_mnist_images(TRAIN_IMAGES_PATH, images))
        ERROR("Reading Training Images");

    // Randomly initialize weights and biases
    Eigen::MatrixXd hidden_weights = Eigen::MatrixXd::Random(NUM_HIDDEN_NEURONS, NUM_INPUT_NEURONS);
    Eigen::MatrixXd hidden_biases = Eigen::MatrixXd::Zero(NUM_HIDDEN_NEURONS, 1);

    Eigen::MatrixXd output_weights = Eigen::MatrixXd::Random(NUM_OUTPUT_NEURONS, NUM_HIDDEN_NEURONS);
    Eigen::MatrixXd output_biases = Eigen::MatrixXd::Zero(NUM_OUTPUT_NEURONS, 1);

    int num_correct = 0;

    int predicted_index, true_index;
    Eigen::VectorXd::Index max_row, max_col;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
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
            o.maxCoeff(&max_row, &max_col);
            predicted_index = max_row;

            // Find the index of the maximum value in the label (should be 1)
            label.maxCoeff(&max_row, &max_col);
            true_index = max_row;

            // Count it as correct if the indices match
            num_correct += (predicted_index == true_index);

            /* Where the learning happens */
            // Backpropagation output -> hidden
            Eigen::MatrixXd delta_o = o - label;
            output_weights += -ETA * delta_o * h.transpose();
            output_biases += -ETA * delta_o;

            // Backpropagation hidden -> input
            Eigen::MatrixXd delta_h = output_weights.transpose() * delta_o;
            delta_h = delta_h.array() * (h.array() * (1 - h.array()));
            hidden_weights += -ETA * delta_h * img.transpose();
            hidden_biases += -ETA * delta_h;
        }

        std::cout << "Epoch: " << epoch << " Accuracy: " << (double)num_correct / images.rows() * 100 << "%" << std::endl;
        num_correct = 0;
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

