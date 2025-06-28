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
}