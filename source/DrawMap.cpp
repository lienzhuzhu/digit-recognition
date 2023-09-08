#include "include/DrawMap.hpp"


void init_map(Grid& map) {
    for (int row_i = 0; row_i < GRID_ROWS; ++row_i) {
        for (int col_i = 0; col_i < GRID_COLS; ++col_i) {
            sf::RectangleShape& curr_cell = map[row_i][col_i];
            curr_cell.setSize(sf::Vector2f(CELL_SIZE, CELL_SIZE));
            curr_cell.setPosition( sf::Vector2f(col_i * CELL_SIZE, row_i * CELL_SIZE) );
            curr_cell.setOutlineColor(BLACK);
            curr_cell.setOutlineThickness(5.f);
            curr_cell.setFillColor(GRAY);
        }
    }
}

void draw_map(Grid& map, sf::RenderWindow& window) {
    for (int row_i = 0; row_i < GRID_ROWS; ++row_i) {
        for (int col_i = 0; col_i < GRID_COLS; ++col_i) {
            window.draw(map[row_i][col_i]);
        }
    }
}
