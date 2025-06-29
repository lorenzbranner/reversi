#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "reversi.cpp"

namespace py = pybind11;

PYBIND11_MODULE(reversi_cpp, m) {
    m.doc() = "Reversi engine C++ bindings";
   
    m.def("get_valid_moves", &get_valid_moves, "Get list of valid moves",
          py::arg("board"), py::arg("player"));
   
    m.def("is_valid_move", &is_valid_move, "Check a single move", 
        py::arg("board"), py::arg("x"), py::arg("y"), py::arg("player"));
   
    m.def("get_next_board", &get_next_board, "Compute next board state",
        py::arg("board"), py::arg("move"), py::arg("player"));

    m.def("game_over", &game_over, "Returns true if the game is over",
        py::arg("board"), py::arg("num_players"));

    m.def("valid_move_player", &valid_move_player, "Returns true if the player has at least one valid move",
        py::arg("board"), py::arg("player"));
}

