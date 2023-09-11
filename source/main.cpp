#include <chrono>

#include "include/DrawMap.hpp"
#include "include/HandleMouse.hpp"
#include "include/LoadData.hpp"
#include <opencv2/opencv.hpp>


int main() {
    sf::RenderWindow window(sf::VideoMode(CELL_SIZE * GRID_COLS + PADDING, CELL_SIZE * GRID_ROWS), "Draw a number", sf::Style::Titlebar | sf::Style::Close);

    Grid map;
    init_map(map);

    Coords cell;


    sf::Font font;
    if (!font.loadFromFile("../res/SpaceMonoNerdFont-Bold.ttf")) {
        return 1;
    }
    sf::Text text;
    text.setFont(font);
    text.setString("Press 'r' key to clear");
    text.setCharacterSize(20);
    text.setFillColor(WHITE);
    text.setPosition(GRID_COLS * CELL_SIZE + (PADDING / 2) - (text.getGlobalBounds().width / 2), 3 * CELL_SIZE);


    Eigen::MatrixXd train_labels = read_mnist_labels(TRAIN_LABELS_PATH);
    Eigen::MatrixXd train_images = read_mnist_images(TRAIN_IMAGES_PATH);


    for (int i = 0; i < 10; ++i) {
        Eigen::MatrixXd label_vector = train_labels.row(i);
        for (int j = 0; j < label_vector.size(); ++j) {
            std::cout << label_vector(j) << " ";
        }
        std::cout << std::endl;
    }


    Eigen::VectorXd single_image_vector = train_images.row(0);
    cv::Mat img(28, 28, CV_64F);
    Eigen::Map<Eigen::MatrixXd>(img.ptr<double>(), img.rows, img.cols) = Eigen::Map<const Eigen::MatrixXd>(single_image_vector.data(), 28, 28);
    cv::Mat img_8u;
    img.convertTo(img_8u, CV_8U, 255.0);
    cv::imshow("Training Image", img_8u);


    while (window.isOpen()) {

        sf::Event event;
        while (window.pollEvent(event))
        {
            switch(event.type) {
                case sf::Event::Closed:
                    window.close();
                    break;
                default:
                    break;
            }
        }

        if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left) || sf::Mouse::isButtonPressed(sf::Mouse::Button::Right)) {
            if (mouse_is_in_grid(window)) {
                cell = get_mouse_cell(window); 

                if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
                    map[cell.second][cell.first].setFillColor(BLACK);
                } else {
                    map[cell.second][cell.first].setFillColor(GRAY);
                }
            }
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::R)) {
            init_map(map);
        }

        sf::Text guess_prompt;
        guess_prompt.setFont(font);
        guess_prompt.setString("You drew the number");
        guess_prompt.setCharacterSize(20);
        guess_prompt.setFillColor(WHITE);
        guess_prompt.setPosition(GRID_COLS * CELL_SIZE + (PADDING / 2) - (guess_prompt.getGlobalBounds().width / 2), 10 * CELL_SIZE);

        window.clear();
        draw_map(map, window);
        window.draw(text);
        window.draw(guess_prompt);
        window.display();
    }

    return 0;
}

