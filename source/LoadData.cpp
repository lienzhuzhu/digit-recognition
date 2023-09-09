#include "include/LoadData.hpp"


std::vector<uint8_t> read_mnist_labels(const std::string& full_path) {
    std::ifstream file(full_path, std::ios::binary);

    if (file.is_open()) {
        int magic_number = 0;
        int num_items = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = __builtin_bswap32(magic_number);

        file.read((char*)&num_items, sizeof(num_items));
        num_items = __builtin_bswap32(num_items);

        std::vector<uint8_t> labels(num_items);
        file.read((char*)labels.data(), num_items);

        return labels;
    } else {
        std::cout << "Cannot read MNIST labels from " << full_path << std::endl;
        return {};
    }
}
 

std::vector<std::vector<uint8_t>> read_mnist_images(const std::string& full_path) {
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

        std::vector<std::vector<uint8_t>> images(num_images, std::vector<uint8_t>(num_rows * num_cols));

        for (int i = 0; i < num_images; ++i) {
            file.read((char*)images[i].data(), num_rows * num_cols);
        }

        return images;
    } else {
        std::cout << "Cannot read MNIST images from " << full_path << std::endl;
        return {};
    }
}
