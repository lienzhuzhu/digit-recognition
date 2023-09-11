#pragma once

#include <SFML/Graphics.hpp>
#include <Eigen/Dense>
#include <unordered_set>
#include <queue>
#include <iostream>
#include <fstream>
#include <vector>

#define GRID_ROWS   28
#define GRID_COLS   28
#define CELL_SIZE   25
#define PADDING     500

#define BLACK sf::Color(4,4,3)
#define GRAY sf::Color(93,115,126)
#define YELLOW sf::Color(255,240,124)
#define WHITE sf::Color(240,247,238)
#define RED sf::Color(255,102,99)
#define BLUE sf::Color(0,109,170)

using Grid = std::array<std::array<sf::RectangleShape, GRID_COLS>, GRID_ROWS>;
using Coords = std::pair<unsigned short, unsigned short>;
