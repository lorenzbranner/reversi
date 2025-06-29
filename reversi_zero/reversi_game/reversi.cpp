// reversi.cpp
#include <vector>
#include <cstdint>
#include <array>
#include <algorithm>
#include <iostream>

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


bool valid_move_player(const std::vector<std::vector<uint8_t>>& board, uint8_t player) {
    for (int y = 0; y < BOARD_SIZE; ++y) {
        for (int x = 0; x < BOARD_SIZE; ++x) {
            if (is_valid_move(board, x, y, player)) {
                return true;
            }
        }
    }
    return false;
}


std::vector<std::vector<uint8_t>> get_next_board(
    const std::vector<std::vector<uint8_t>>& board,
    std::pair<int, int> move,
    uint8_t player
) {
    auto new_board = board;
    int x = move.first;
    int y = move.second;
    new_board[y][x] = player;

    for (const auto& [dx, dy] : directions) {
        std::vector<std::pair<int, int>> path;
        int cx = x + dx;
        int cy = y + dy;
        while (cx >= 0 && cx < BOARD_SIZE && cy >= 0 && cy < BOARD_SIZE) {
            uint8_t val = new_board[cy][cx];
            if (val == 0 || val == BLOCKED) break;
            if (val == player) {
                for (const auto& [px, py] : path) {
                    new_board[py][px] = player;
                }
                break;
            }
            path.emplace_back(cx, cy);
            cx += dx;
            cy += dy;
        }
    }

    return new_board;
}


bool game_over(const std::vector<std::vector<uint8_t>>& board, uint8_t num_players) {
    constexpr int BOARD_SIZE = 15;
    constexpr uint8_t BLOCKED = 5;

    for (int y = 0; y < BOARD_SIZE; ++y) {
        for (int x = 0; x < BOARD_SIZE; ++x) {
            if (board[y][x] != 0 && board[y][x] != 5) continue;

            for (uint8_t player = 1; player <= num_players; ++player) {
                if (is_valid_move(board, x, y, player)){
                    return false;
                }
            }
        }
    }

    return true; 
}
