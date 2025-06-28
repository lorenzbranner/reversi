// reversi.cpp
#include <vector>
#include <cstdint>
#include <array>
#include <algorithm>

constexpr int BOARD_SIZE = 15;
constexpr uint8_t BLOCKED = 5;

std::array<std::pair<int, int>, 8> directions = {
    {{-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}}
};

bool validate_direction(const std::vector<std::vector<uint8_t>>& board, int x, int y, int dx, int dy, uint8_t player) {
    int cx = x + dx, cy = y + dy;
    bool found_enemy = false;
    while (cx >= 0 && cx < BOARD_SIZE && cy >= 0 && cy < BOARD_SIZE) {
        uint8_t val = board[cy][cx];
        if (val == 0 || val == BLOCKED) return false;
        if (val == player) return found_enemy;
        found_enemy = true;
        cx += dx;
        cy += dy;
    }
    return false;
}

bool is_valid_move(const std::vector<std::vector<uint8_t>>& board, int x, int y, uint8_t player) {
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) return false;
    if (board[y][x] != 0) return false;
    for (const auto& [dx, dy] : directions) {
        if (validate_direction(board, x, y, dx, dy, player)) return true;
    }
    return false;
}

std::vector<std::pair<int, int>> get_valid_moves(const std::vector<std::vector<uint8_t>>& board, uint8_t player) {
    std::vector<std::pair<int, int>> moves;
    for (int y = 0; y < BOARD_SIZE; ++y) {
        for (int x = 0; x < BOARD_SIZE; ++x) {
            if (is_valid_move(board, x, y, player)) {
                moves.emplace_back(x, y);
            }
        }
    }
    return moves;
}
