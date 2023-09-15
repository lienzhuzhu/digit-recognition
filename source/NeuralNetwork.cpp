#include "include/NeuralNetwork.hpp"


#define NUM_INPUT_NEURONS   784
#define NUM_HIDDEN_NEURONS  20
#define NUM_OUTPUT_NEURONS  10


void print_dimensions(const std::string& name, const Eigen::MatrixXd& mat) {
    std::cout << std::left << std::setw(20) << name << mat.rows() << " x " << mat.cols() << std::endl;
}


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
    Eigen::VectorXd img, label;
    Eigen::VectorXd hidden_preactivation, hidden_activation;
    Eigen::VectorXd output_preactivation, output_activation;
    
    if (read_mnist_labels(TRAIN_LABELS_PATH, labels))
        ERROR("Reading Training Labels");
    if (read_mnist_images(TRAIN_IMAGES_PATH, images))
        ERROR("Reading Training Images");


    Eigen::MatrixXd hidden_weights = Eigen::MatrixXd::Random(NUM_HIDDEN_NEURONS, NUM_INPUT_NEURONS);
    Eigen::MatrixXd hidden_biases = Eigen::MatrixXd::Zero(NUM_HIDDEN_NEURONS, 1);

    Eigen::MatrixXd output_weights = Eigen::MatrixXd::Random(NUM_OUTPUT_NEURONS, NUM_HIDDEN_NEURONS);
    Eigen::MatrixXd output_biases = Eigen::MatrixXd::Zero(NUM_OUTPUT_NEURONS, 1);


    int num_correct = 0;
    int prediction, true_index;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::cout << "Training Epoch: " << epoch << std::endl;

        for (int i = 0; i < labels.rows(); ++i) {
            img = images.row(i).transpose();
            label = labels.row(i).transpose();

            hidden_preactivation = hidden_weights * img + hidden_biases;
            hidden_activation = sigmoid(hidden_preactivation);

            output_preactivation = output_weights * hidden_activation + output_biases;
            output_activation = sigmoid(output_preactivation);

            output_activation.maxCoeff(&prediction);
            label.maxCoeff(&true_index);
            
            num_correct += (prediction == true_index);

            /* Where the learning happens */
            Eigen::MatrixXd delta_output = output_activation - label;                                               // 10 x 1   // output_gradient
            output_weights += -ETA * delta_output * hidden_activation.transpose();                                  // 10 x 20  // weights_gradient = np.dot(output_gradient, self.input.T)
                                                                                                                                // self.weights -= learning_rate * weights_gradient        
            output_biases += -ETA * delta_output;                                                                   // 10 x 1   // self.bias -= learning_rate * output_gradient

            Eigen::MatrixXd delta_hidden = output_weights.transpose() * delta_output;                               // 20 x 1   // input_gradient = np.dot(self.weights.T, output_gradient)
            delta_hidden = delta_hidden.array() * (hidden_activation.array() * (1 - hidden_activation.array()));    // 20 x 1
            hidden_weights += -ETA * delta_hidden * img.transpose();                                                // 20 x 784
            hidden_biases += -ETA * delta_hidden;                                                                   // 20 x 1
        }

        std::cout << "Epoch: " << epoch << " Accuracy: " << (double)num_correct / images.rows() * 100 << "%" << std::endl;
        num_correct = 0;
    }

    save_parameters(hidden_weights, "model/hidden_weights.txt");
    save_parameters(hidden_biases, "model/hidden_biases.txt");
    save_parameters(output_weights, "model/output_weights.txt");
    save_parameters(output_biases, "model/output_biases.txt");

    return SUCCESS;
}

