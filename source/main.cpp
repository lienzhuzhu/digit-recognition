#include <chrono>
#include <iostream>

#include "include/DrawMap.hpp"
#include "include/HandleMouse.hpp"


int main() {
    sf::RenderWindow window(sf::VideoMode(CELL_SIZE * GRID_COLS + PADDING, CELL_SIZE * GRID_ROWS), "Draw a number", sf::Style::Titlebar | sf::Style::Close);

    Grid map;
    init_map(map);

    Coords cell;

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

        window.clear();
        draw_map(map, window);
        window.display();
    }

    return 0;
}

