#include <chrono>

#include "include/DrawMap.hpp"
#include "include/HandleMouse.hpp"
#include "include/NeuralNetwork.hpp"

#ifdef  DEBUG
#include <opencv2/opencv.hpp>
#endif



int main() {
    sf::RenderWindow window(sf::VideoMode(CELL_SIZE * GRID_COLS + PADDING, CELL_SIZE * GRID_ROWS), "Draw a number", sf::Style::Titlebar | sf::Style::Close);

    Grid map;
    init_map(map);

    Coords cell;


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


#ifdef  TRAIN_MODE
    if (train_nn()) {
        std::cout << "training the model failed" << std::endl;
        return FAILURE;
    }
#endif

    Eigen::MatrixXd w_i_h, b_i_h, w_h_o, b_h_o;

    if (load_parameters("model/w_i_h.txt", w_i_h)) {
        std::cerr << "Could not load w_i_h" << std::endl;
        return FAILURE;
    }
    if (load_parameters("model/b_i_h.txt", b_i_h)) {
        std::cerr << "Could not load b_i_h" << std::endl;
        return FAILURE;
    }
    if (load_parameters("model/w_h_o.txt", w_h_o)) {
        std::cerr << "Could not load w_h_o" << std::endl;
        return FAILURE;
    }
    if (load_parameters("model/b_h_o.txt", b_h_o)) {
        std::cerr << "Could not load b_h_o" << std::endl;
        return FAILURE;
    }

    if (test_nn(w_i_h, b_i_h, w_h_o, b_h_o)) {
        std::cout << "testing failed" << std::endl;
        return FAILURE;
    }

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

