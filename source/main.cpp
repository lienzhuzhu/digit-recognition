#include <chrono>

#include "include/DrawMap.hpp"
#include "include/HandleMouse.hpp"
#include "include/NeuralNetwork.hpp"

#ifdef  DEBUG
#include <opencv2/opencv.hpp>
#endif


int main() {

#ifdef  TRAIN_MODE
    if (train_nn()) {
        std::cout << "training the model failed" << std::endl;
        return FAILURE;
    }
#endif


    Eigen::MatrixXd hidden_weights, hidden_biases, output_weights, output_biases;

    if (load_parameters("model/w_i_h.txt", hidden_weights))
        ERROR("hidden weights file");

    if (load_parameters("model/b_i_h.txt", hidden_biases))
        ERROR("hidden biases file");

    if (load_parameters("model/w_h_o.txt", output_weights))
        ERROR("output weights file");

    if (load_parameters("model/b_h_o.txt", output_biases))
        ERROR("output biases file");

    if (test_nn(hidden_weights, hidden_biases, output_weights, output_biases))
        ERROR("test_nn()");


    /*
    sf::RenderWindow window(sf::VideoMode(CELL_SIZE * GRID_COLS + PADDING, CELL_SIZE * GRID_ROWS), "Draw a number", sf::Style::Titlebar | sf::Style::Close);

    Grid map;
    init_map(map);
    Coords cell;

    bool make_prediction = false;


    sf::Font font;
    if (!font.loadFromFile("../res/SpaceMonoNerdFont-Bold.ttf")) {
        return FAILURE;
    }
    sf::Text text;
    text.setFont(font);
    text.setString("Press 'r' key to clear");
    text.setCharacterSize(20);
    text.setFillColor(WHITE);
    text.setPosition(GRID_COLS * CELL_SIZE + (PADDING / 2) - (text.getGlobalBounds().width / 2), 3 * CELL_SIZE);

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
                make_prediction = true;
            }
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::R)) {
            init_map(map);
        }

        sf::Text guess_prompt;
        guess_prompt.setFont(font);
        guess_prompt.setCharacterSize(20);
        guess_prompt.setFillColor(WHITE);

        if (make_prediction) {
            int prediction = predict( get_drawing(map), w_i_h, b_i_h, w_h_o, b_h_o );
            std::string prompt = "You drew the number " + std::to_string(prediction);
            guess_prompt.setString(prompt);
            guess_prompt.setPosition(GRID_COLS * CELL_SIZE + (PADDING / 2) - (guess_prompt.getGlobalBounds().width / 2), 10 * CELL_SIZE);
        }

        window.clear();
        draw_map(map, window);
        window.draw(text);
        window.draw(guess_prompt);
        window.display();

    }
    */

    return 0;
}

