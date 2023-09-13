#pragma once

#include "global.hpp"


void init_map(Grid&);
void draw_map(Grid&, sf::RenderWindow&);
Eigen::MatrixXd get_drawing(Grid &map);
