#include "include/LoadData.hpp"


ReturnStatus read_mnist_labels(const std::string &full_path, Eigen::MatrixXd &labels) {
    std::ifstream file(full_path, std::ios::binary);

    if (file.is_open()) {
        int magic_number = 0;
        int num_items = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = __builtin_bswap32(magic_number);

        file.read((char*)&num_items, sizeof(num_items));
        num_items = __builtin_bswap32(num_items);

        labels = Eigen::MatrixXd::Zero(num_items, 10);

        for (int i = 0; i < num_items; ++i) {
            uint8_t label;
            file.read((char*)&label, 1);
            labels(i, label) = 1.0;
        }

        return SUCCESS;
    } else {
        std::cout << "Cannot read MNIST labels from " << full_path << std::endl;
        return FAILURE;
    }
}


ReturnStatus read_mnist_images(const std::string &full_path, Eigen::MatrixXd &images) {
    std::ifstream file(full_path, std::ios::binary);

    if (file.is_open()) {
        int magic_number = 0;
        int num_images = 0;
        int num_rows = 0;
        int num_cols = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = __builtin_bswap32(magic_number);

        file.read((char*)&num_images, sizeof(num_images));
        num_images = __builtin_bswap32(num_images);

        file.read((char*)&num_rows, sizeof(num_rows));
        num_rows = __builtin_bswap32(num_rows);

        file.read((char*)&num_cols, sizeof(num_cols));
        num_cols = __builtin_bswap32(num_cols);

        images.resize(num_images, num_rows * num_cols);

        for (int i = 0; i < num_images; ++i) {
            for (int j = 0; j < num_rows * num_cols; ++j) {
                uint8_t pixel;
                file.read((char*)&pixel, 1);
                images(i, j) = static_cast<double>(pixel) / 255.0;
            }
        }

        return SUCCESS;
    } else {
        std::cout << "Cannot read MNIST images from " << full_path << std::endl;
        return FAILURE;
    }
}
